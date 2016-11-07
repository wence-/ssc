CFLAGS = -I.

include ${PETSC_DIR}/lib/petsc/conf/variables

HDR = libssc.h

SRC = libssc.c

OBJ = libssc.o

LIBNAME = libssc

LIB = $(LIBNAME).$(SL_LINKER_SUFFIX)

all: $(LIB)

clean:
	${RM} ${LIB} ${OBJ}
	${RM} -rf ${LIB}.dSYM

ARCH = $(shell uname)

$(SRC): $(HDR)

$(OBJ): $(SRC)
	${PETSC_COMPILE_SINGLE} $<

$(LIB): $(OBJ)

ifeq ($(ARCH), Linux)
  $(LIB):
	${CLINKER} -shared -Wl,-soname,${LIBNAME}.${SL_LINKER_SUFFIX} -o ${LIB} ${OBJ} ${PETSC_LIB} ${OTHERSHAREDLIBS}	
else ifeq ($(ARCH), Darwin)
  $(LIB): 
	export MACOSX_DEPLOYMENT_TARGET=`sw_vers -productVersion | cut -d . -f1,2`; \
        ${LD_SHARED} -g -dynamiclib -single_module -multiply_defined suppress -undefined dynamic_lookup ${DARWIN_COMMONS_USE_DYLIBS} -o ${LIB} ${OBJ} -L${PETSC_LIB_DIR} ${PETSC_LIB} ${OTHERSHAREDLIBS} ${SL_LINKER_LIBS} -lm -lc; \
        ${DSYMUTIL} ${LIB}; \
        install_name_tool -id $(abspath $(LIB)) $(LIB)
else
  $(LIB): unknown_target
endif

