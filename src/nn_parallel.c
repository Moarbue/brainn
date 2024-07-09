#include "../include/nn.h"

#include <pthread.h>
#include <stdbool.h>

typedef struct {
    NN nn;
    Mat ti;
    Mat to;
    bsize bs;
    bsize e;
    bsize i;
    bfloat l;
    bool finished;
} TData;

pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
pthread_mutex_t mutex   = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  cv      = PTHREAD_COND_INITIALIZER;

void nn_palloc(NN *nn, bsize nthreads, bsize othreads)
{
    if (nthreads == 0) PANIC("nn_palloc(): Number of threads cannot be zero!");

    bsize *arch, layers;
    nn_get_arch(*nn, &arch, &layers);

    for (bsize n = 1; n < othreads; n++) {
        bsize c = n * layers;
        bsize d = n * nn->l;

        vec_free(nn->a[c]);
        for (bsize l = 0; l < nn->l; l++) {
            vec_free(nn->a [c + l+1]);
            vec_free(nn->z [d + l]);
            vec_free(nn->da[d + l]);
            vec_free(nn->gb[d + l]);
            mat_free(nn->gw[d + l]);
        }
    }

    nn->a  = (Vec *) BREALLOC(nn->a,  nthreads * layers * sizeof (Vec));
    nn->z  = (Vec *) BREALLOC(nn->z,  nthreads * nn->l  * sizeof (Vec));
    nn->da = (Vec *) BREALLOC(nn->da, nthreads * nn->l  * sizeof (Vec));
    nn->gb = (Vec *) BREALLOC(nn->gb, nthreads * nn->l  * sizeof (Vec));
    nn->gw = (Mat *) BREALLOC(nn->gw, nthreads * nn->l  * sizeof (Mat));
    nn->gc = (bsize *) BREALLOC(nn->gc, nthreads * sizeof (bsize));

    if (nn->a == NULL || nn->z == NULL || nn->da == NULL || nn->gb == NULL || nn->gw == NULL || nn->gc == NULL) 
        PANIC("nn_palloc(): Failed to reallocate memory!");

    arch++;

    for (bsize n = 1; n < nthreads; n++) {
        bsize c = n * layers;
        bsize d = n * nn->l;

        nn->a[c] = vec_alloc(*(arch-1));
        for (bsize l = 0; l < nn->l; l++) {
            nn->a [c + l+1] = vec_alloc(arch[l]);
            nn->z [d + l]   = vec_alloc(arch[l]);
            nn->da[d + l]   = vec_alloc(arch[l]);
            nn->gb[d + l]   = vec_alloc(arch[l]);
            nn->gw[d + l]   = mat_alloc(arch[l], arch[l-1]);
        }
        nn->gc[n] = 0;
    }

    BFREE(--arch);
}

void nn_pinit(NN nn, bsize nthreads)
{
    for (bsize n = 0; n < nthreads; n++) {
        bsize c = n * (nn.l+1);
        bsize d = n * nn.l;

        vec_fill(nn.a[c], 0);
        for (bsize l = 0; l < nn.l; l++) {
            vec_fill(nn.a [c + l], 0);
            vec_fill(nn.z [d + l], 0);
            vec_fill(nn.da[d + l], 0);
            vec_fill(nn.gb[d + l], 0);
            mat_fill(nn.gw[d + l], 0);
        }
    }
}

void nn_pgradient(NN nn, bsize nthreads)
{
    for (bsize n = 1; n < nthreads; n++) {
        bsize d = n * nn.l;

        for (bsize l = 0; l < nn.l; l++) {
            for (bsize r = 0; r < nn.gw[l].r; r++) {
                vec_el(nn.gb[l], r) += vec_el(nn.gb[d + l], r);
                vec_el(nn.gb[d + l], r) = 0;
                for (bsize c = 0; c < nn.gw[l].c; c++) {
                    mat_el(nn.gw[l], r, c) += mat_el(nn.gw[d + l], r, c);
                    mat_el(nn.gw[d + l], r, c) = 0;
                }
            }
        }
        nn.gc[0] += nn.gc[n];
        nn.gc[n] = 0;
    }
}

void *process_batch(void *arg)
{
    TData *td = (TData *) arg;

    td->nn.a  += td->i * (td->nn.l+1);
    td->nn.z  += td->i * td->nn.l;
    td->nn.da += td->i * td->nn.l;
    td->nn.gb += td->i * td->nn.l;
    td->nn.gw += td->i * td->nn.l;
    td->nn.gc += td->i;

    for (bsize e = 0; e < td->e; e++) {
        // perform a single forward-backward-pass
        pthread_rwlock_rdlock(&rwlock);

        for (bsize b = 0; b < td->bs; b++) {
            nn_forward(td->nn, mat_to_row_vec(td->ti, b));
            nn_backpropagate(td->nn, mat_to_row_vec(td->to, b));
        }
        
        td->l = nn_loss(td->nn, td->ti, td->to) * td->ti.r;
        td->finished = true;

        pthread_rwlock_unlock(&rwlock);

        // wait for main thread to finish optimizing
        pthread_mutex_lock(&mutex);
        while(td->finished) pthread_cond_wait(&cv, &mutex);
        pthread_mutex_unlock(&mutex);
    }

    return NULL;
}

void nn_ptrain(NN *nn, Mat ti, Mat to, bsize batch_size, bsize epochs, bsize nthreads, int report_loss)
{
    if (nthreads == 1) {
        nn_train(*nn, ti, to, batch_size, epochs, report_loss);
        return;
    }

    Mat m = {.r = ti.r, .c = ti.c + to.c, .e = ti.e, .s = 0};

    nn_palloc(nn, nthreads, 1);
    nn_pinit(*nn, nthreads);

    pthread_t *threads = (pthread_t *) BALLOC(nthreads * sizeof (pthread_t));
    if (threads == NULL) PANIC("nn_ptrain(): Failed to allocate memory for threads!");

    TData *td = (TData *) BALLOC(nthreads * sizeof (TData));
    if (td == NULL) PANIC("nn_ptrain(): Failed to allocate memory for training data!");

    if (nthreads > ti.r) PANIC("nn_ptrain(): nthreads greater than training samples: %zu > %zu", nthreads, ti.r);

    bsize cs = ti.r / nthreads; // chunk size
    bsize r  = ti.r % nthreads; // remainder
    batch_size /= nthreads;
    
    if (batch_size > cs) batch_size = cs;

    for (bsize n = 0, i = 0; n < nthreads; n++, i += cs, cs = ti.r / nthreads) {
        if (r > 0) {
            r--;
            cs++;
        }

        td[n] = (TData) {
            .nn = *nn,
            .ti = mat_sub_mat(ti, i, 0, cs, ti.c),
            .to = mat_sub_mat(to, i, 0, cs, to.c),
            .bs = batch_size,
            .e  = epochs,
            .i  = n,
            .l  = 0,
            .finished = false,
        };

        pthread_create(&threads[n], NULL, process_batch, (void *) &td[n]);
    }

    for (bsize e = 0; e < epochs+1; e++) {
        mat_shuffle_rows(m);

        // wait for other threads to finish
        bool finished = false;
        while (!finished) {
            for (bsize n = 0; n < nthreads; n++) {
                finished = td[n].finished;
                if (!finished) break;
            }
        }

        // optimize network
        pthread_rwlock_wrlock(&rwlock);

        bfloat loss = 0.f;
        for (bsize n = 0; n < nthreads; n++) loss += td[n].l;
        loss /= (bfloat) ti.r;
        if (report_loss) printf("E: %zu L: %.5f\r", e, loss);

        nn_pgradient(*nn, nthreads);
        nn_evolve(*nn);

        pthread_rwlock_unlock(&rwlock);

        // tell other networks that optimizing is finished
        pthread_mutex_lock(&mutex);
        for (bsize n = 0; n < nthreads; n++) td[n].finished = false;
        pthread_cond_broadcast(&cv);
        pthread_mutex_unlock(&mutex);
    }
    if (report_loss) printf("\n");

    for (bsize n = 0; n < nthreads; n++) {
        char *c;
        pthread_join(threads[n], (void **)&c);
    }

    nn_palloc(nn, 1, nthreads);
    BFREE(threads);
    BFREE(td);
}