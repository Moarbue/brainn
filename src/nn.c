#include "../include/nn.h"

NN nn_alloc(bsize *arch, bsize layers)
{
    if (layers < 1) PANIC("nn_alloc(): Minimum layer count is 1!");

    NN nn;

    nn.l  = layers - 1;
    nn.a  = (Vec*) BALLOC(layers * sizeof (Vec));
    nn.z  = (Vec*) BALLOC(nn.l   * sizeof (Vec));
    nn.b  = (Vec*) BALLOC(nn.l   * sizeof (Vec));
    nn.w  = (Mat*) BALLOC(nn.l   * sizeof (Mat));
    nn.da = (Vec*) BALLOC(nn.l   * sizeof (Vec));
    nn.gb = (Vec*) BALLOC(nn.l   * sizeof (Vec));
    nn.gw = (Mat*) BALLOC(nn.l   * sizeof (Mat));
    nn.gc = (bsize*) BALLOC(sizeof (bsize));
    nn.o  = (Optimizer*) BALLOC(sizeof (Optimizer));

    if (nn.a  == NULL || nn.z  == NULL || nn.b  == NULL || nn.w  == NULL ||
        nn.da == NULL || nn.gb == NULL || nn.gw == NULL || nn.gc == NULL || nn.o == NULL)
        PANIC("nn_alloc(): Failed to allocate memory!");

    nn_input(nn) = vec_alloc(arch[0]);
    arch++;

    for (bsize l = 0; l < nn.l; l++) {
        nn.a [l+1] = vec_alloc(arch[l]);
        nn.z [l]   = vec_alloc(arch[l]);
        nn.b [l]   = vec_alloc(arch[l]);
        nn.w [l]   = mat_alloc(arch[l], arch[l-1]);
        nn.da[l]   = vec_alloc(arch[l]);
        nn.gb[l]   = vec_alloc(arch[l]);
        nn.gw[l]   = mat_alloc(arch[l], arch[l-1]);
    }
    *nn.gc = 0;

    nn.hf  = Sigmoid;
    nn.dhf = dSigmoid;
    nn.of  = Sigmoid;
    nn.dof = dSigmoid;

    nn.C   = SEL;
    nn.dC  = dSEL;

    *nn.o  = optimizer_SGD(1e-2);

    return nn;
}

void nn_init(NN nn, bfloat min, bfloat max)
{
    vec_fill(nn_input(nn), 0);
    for (bsize l = 0; l < nn.l; l++) {
        vec_fill(nn.a [l+1], 0);
        vec_fill(nn.z [l],   0);
        vec_fill(nn.b [l],   0);
        mat_rand(nn.w [l], min, max);
        vec_fill(nn.da[l],   0);
        vec_fill(nn.gb[l],   0);
        mat_fill(nn.gw[l],   0);
    }
}

Vec nn_forward(NN nn, Vec input)
{
    vec_copy(nn_input(nn), input);
    for (bsize l = 0; l < nn.l; l++) {
        vec_mat_mul(nn.a[l+1], nn.a[l], nn.w[l]);
        vec_sum(nn.a[l+1], nn.b[l]);
        vec_copy(nn.z[l], nn.a[l+1]);
        
        if ((l+1) == nn.l) vec_activate(nn.a[l+1], nn.of);
        else               vec_activate(nn.a[l+1], nn.hf);
    }

    return nn_output(nn);
}

void nn_backpropagate(NN nn, Vec output)
{
    if (output.c != nn_output(nn).c) PANIC("Network output size differs from expected output size! %zu != %zu", output.c, nn_output(nn).c);

    bsize L, N, P;  // layers, neurons, neurons in previous layer
    bfloat dCo;     // derivative of Loss function in respect to neuron output
    bfloat doz;     // derivative of activation function in respect to weighted input

    // clean intermediate values
    L = nn.l-1;
    for (bsize l = 0; l < L; l++) {
        vec_fill(nn.da[l], 0);
    }

    // calculate derivative of cost function
    N = nn.da[L].c;
    for (bsize n = 0; n < N; n++) {
        vec_el(nn.da[L], n) = (nn.dC)(vec_el(output, n), vec_el(nn_output(nn), n));
    }

    for (bsize l = L; l <= L; l--) {
        N = nn.a[l+1].c;
        P = nn.a[l].c;
        for (bsize n = 0; n < N; n++) { 
            dCo = vec_el(nn.da[l], n);
            doz = (l == L) ? (nn.dof)(vec_el(nn.z[l], n)) : (nn.dhf)(vec_el(nn.z[l], n));

            vec_el(nn.gb[l], n) += dCo * doz;
            for (bsize p = 0; p < P; p++) {
                mat_el(nn.gw[l], n, p) += dCo * doz * vec_el(nn.a[l], p);
                if (l == 0) continue; // skip input layer
                vec_el(nn.da[l-1], p)  += dCo * doz * mat_el(nn.w[l], n, p);
            }
        }
    }

    (*nn.gc)++;
}

void nn_evolve(NN nn)
{
    bsize L, N, P;  // layers, neurons, neurons in previous layer

    L = nn.l;
    for (bsize l = 0; l < L; l++) {
        N = nn.w[l].r;
        P = nn.w[l].c;
        for (bsize n = 0; n < N; n++) {
            vec_el(nn.gb[l], n) /= (bfloat)(*nn.gc);
            vec_el(nn.b [l], n)  = optimizer_update_bias(nn.o, vec_el(nn.b[l], n), vec_el(nn.gb[l], n), l, n);
            vec_el(nn.gb[l], n)  = 0;
            for (bsize p = 0; p < P; p++) {
                mat_el(nn.gw[l], n, p) /= (bfloat)(*nn.gc);
                mat_el(nn.w [l], n, p)  = optimizer_update_weight(nn.o, mat_el(nn.w[l], n, p), mat_el(nn.gw[l], n, p), l, n, p);
                mat_el(nn.gw[l], n, p)  = 0;
            }
        }
    }
    *nn.gc = 0;
}

bfloat nn_loss(NN nn, Mat training_inputs, Mat training_outputs)
{
    bfloat cost;
    Vec input, output;
    bsize samples;

    samples = training_inputs.r;
    cost = 0;
    for (bsize i = 0; i < samples; i++) {
        input  = mat_to_row_vec(training_inputs,  i);
        output = mat_to_row_vec(training_outputs, i);

        if (output.c != nn_output(nn).c) PANIC("Network output size differs from expected output size! %zu != %zu", output.c, nn_output(nn).c);

        nn_forward(nn, input);
        for (bsize j = 0; j < output.c; j++) {
            cost += (nn.C)(vec_el(output, j), vec_el(nn_output(nn), j));
        }
    }
    cost /= (bfloat)samples;

    return cost;
}

void nn_train(NN nn, Mat ti, Mat to, bsize batch_size, bsize epochs, int report_loss)
{
    if (batch_size > ti.r) batch_size = ti.r;

    Mat m = {.r = ti.r, .c = ti.c + to.c, .e = ti.e, .s = 0};

    for (bsize e = 0; e < epochs; e++) {
        mat_shuffle_rows(m);

        for (bsize b = 0; b < batch_size; b++) {
            nn_forward(nn, mat_to_row_vec(ti, b));
            nn_backpropagate(nn, mat_to_row_vec(to, b));
        }
        nn_evolve(nn);
        if (report_loss) printf("E: %zu L: %.5f\r", e, nn_loss(nn, ti, to));
    } 
    if (report_loss) printf("\n");
}

void nn_free(NN nn)
{
    vec_free(nn_input(nn));
    for (bsize l = 0; l < nn.l; l++) {
        vec_free(nn.a[l+1]);
        vec_free(nn.z[l]);
        vec_free(nn.b[l]);
        mat_free(nn.w[l]);
        vec_free(nn.da[l]);
        vec_free(nn.gb[l]);
        mat_free(nn.gw[l]);
    }

    optimizer_free(nn.o, nn.l);

    BFREE(nn.a);
    BFREE(nn.z);
    BFREE(nn.b);
    BFREE(nn.w);
    BFREE(nn.da);
    BFREE(nn.gb);
    BFREE(nn.gw);
    BFREE(nn.gc);
    BFREE(nn.o);
}


void nn_print_intern(NN nn, const char *name)
{
    char buf[32];
    printf("%s = [\n", name);
    for (bsize i = 0; i < nn.l; i++) {
        snprintf(buf, sizeof (buf), "w[%zu]", i);
        mat_print_intern(nn.w[i], buf, 4);
        snprintf(buf, sizeof (buf), "b[%zu]", i);
        vec_print_intern(nn.b[i], buf, 4);
    }
    printf("]\n");
}