include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

LIB = $(call SONAME_FUNCTION,libssc,1)
LINKER_FLAGS = $(call SL_LINKER_FUNCTION,libssc,1,1)

all: $(LIB)
$(LIB): libssc.c libssc.h
	${CC} ${LINKER_FLAGS} -o ${LIB} ${CC_FLAGS} -O0 libssc.c ${PETSC_LIB} ${PETSC_CC_INCLUDES} -I.
