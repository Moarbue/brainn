#include "../config.h"

#ifndef _BRAINN_LOSS_H_
#define _BRAINN_LOSS_H_

typedef bfloat (loss_function)(bfloat, bfloat);
typedef bfloat (dloss_function)(bfloat, bfloat);

bfloat SEL(bfloat t, bfloat y);
bfloat CEL(bfloat t, bfloat y);

bfloat dSEL(bfloat t, bfloat y);
bfloat dCEL(bfloat t, bfloat y);


#endif // _BRAINN_LOSS_H_