set(SEQUENCE_PROCESSING_LOCAL_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SEQUENCE_PROCESSING_C_LOCAL_COMPUTATIONAL_GRAPH_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_CORPUS_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src/Dictionary
        ${SEQUENCE_PROCESSING_C_LOCAL_MATH_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_UTIL_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_REGULAR_ROOT}/src)

file(GLOB_RECURSE SEQUENCE_PROCESSING_LOCAL_DATA_STRUCTURE_SOURCES
        CONFIGURE_DEPENDS
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src/*.c)
add_library(sequence_processing_data_structure STATIC ${SEQUENCE_PROCESSING_LOCAL_DATA_STRUCTURE_SOURCES})
target_include_directories(sequence_processing_data_structure
        PUBLIC
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src)

file(GLOB_RECURSE SEQUENCE_PROCESSING_LOCAL_UTIL_SOURCES
        CONFIGURE_DEPENDS
        ${SEQUENCE_PROCESSING_C_LOCAL_UTIL_ROOT}/src/*.c)
add_library(sequence_processing_util STATIC ${SEQUENCE_PROCESSING_LOCAL_UTIL_SOURCES})
target_include_directories(sequence_processing_util
        PUBLIC
        ${SEQUENCE_PROCESSING_C_LOCAL_UTIL_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src)
target_link_libraries(sequence_processing_util
        PUBLIC
        sequence_processing_data_structure)

file(GLOB_RECURSE SEQUENCE_PROCESSING_LOCAL_MATH_SOURCES
        CONFIGURE_DEPENDS
        ${SEQUENCE_PROCESSING_C_LOCAL_MATH_ROOT}/src/*.c)
add_library(sequence_processing_math STATIC ${SEQUENCE_PROCESSING_LOCAL_MATH_SOURCES})
target_include_directories(sequence_processing_math
        PUBLIC
        ${SEQUENCE_PROCESSING_C_LOCAL_MATH_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_UTIL_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src)
target_link_libraries(sequence_processing_math
        PUBLIC
        sequence_processing_util
        sequence_processing_data_structure
        m)

file(GLOB_RECURSE SEQUENCE_PROCESSING_LOCAL_DICTIONARY_SOURCES
        CONFIGURE_DEPENDS
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src/*.c)
add_library(sequence_processing_dictionary STATIC ${SEQUENCE_PROCESSING_LOCAL_DICTIONARY_SOURCES})
target_include_directories(sequence_processing_dictionary
        PUBLIC
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src/Dictionary
        ${SEQUENCE_PROCESSING_C_LOCAL_MATH_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_UTIL_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src)
target_link_libraries(sequence_processing_dictionary
        PUBLIC
        sequence_processing_math
        sequence_processing_data_structure
        m)

file(GLOB_RECURSE SEQUENCE_PROCESSING_LOCAL_REGULAR_SOURCES
        CONFIGURE_DEPENDS
        ${SEQUENCE_PROCESSING_C_LOCAL_REGULAR_ROOT}/src/*.c)
add_library(sequence_processing_regular STATIC ${SEQUENCE_PROCESSING_LOCAL_REGULAR_SOURCES})
target_include_directories(sequence_processing_regular
        PUBLIC
        ${SEQUENCE_PROCESSING_C_LOCAL_REGULAR_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src/Dictionary
        ${SEQUENCE_PROCESSING_C_LOCAL_MATH_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_UTIL_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src)
target_link_libraries(sequence_processing_regular
        PUBLIC
        sequence_processing_dictionary)

file(GLOB_RECURSE SEQUENCE_PROCESSING_LOCAL_CORPUS_SOURCES
        CONFIGURE_DEPENDS
        ${SEQUENCE_PROCESSING_C_LOCAL_CORPUS_ROOT}/src/*.c)
add_library(sequence_processing_corpus STATIC ${SEQUENCE_PROCESSING_LOCAL_CORPUS_SOURCES})
target_include_directories(sequence_processing_corpus
        PUBLIC
        ${SEQUENCE_PROCESSING_C_LOCAL_CORPUS_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_REGULAR_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src/Dictionary
        ${SEQUENCE_PROCESSING_C_LOCAL_MATH_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_UTIL_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src)
target_link_libraries(sequence_processing_corpus
        PUBLIC
        sequence_processing_regular)

file(GLOB_RECURSE SEQUENCE_PROCESSING_LOCAL_COMPUTATIONAL_GRAPH_SOURCES
        CONFIGURE_DEPENDS
        ${SEQUENCE_PROCESSING_C_LOCAL_COMPUTATIONAL_GRAPH_ROOT}/src/*.c)
add_library(sequence_processing_computational_graph STATIC ${SEQUENCE_PROCESSING_LOCAL_COMPUTATIONAL_GRAPH_SOURCES})
target_include_directories(sequence_processing_computational_graph
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SEQUENCE_PROCESSING_C_LOCAL_COMPUTATIONAL_GRAPH_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_MATH_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src)
target_link_libraries(sequence_processing_computational_graph
        PUBLIC
        sequence_processing_math
        sequence_processing_data_structure
        m)

add_library(SequenceProcessing
        Classification/BorrowedFunctionProxy.c
        Classification/BorrowedFunctionProxy.h
        Classification/GatedRecurrentUnitModel.c
        Classification/GatedRecurrentUnitModel.h
        Classification/JavaRandomCompat.c
        Classification/JavaRandomCompat.h
        Classification/LongShortTermMemoryModel.c
        Classification/LongShortTermMemoryModel.h
        Classification/RecurrentModelGraphBridge.c
        Classification/RecurrentModelGraphBridge.h
        Classification/RecurrentNeuralNetworkModel.c
        Classification/RecurrentNeuralNetworkModel.h
        Classification/Transformer.c
        Classification/Transformer.h
        Classification/TransformerTokenStore.c
        Classification/TransformerTokenStore.h
        SequenceProcessing.c
        SequenceProcessing.h
        Functions/AdditionByConstant.c
        Functions/AdditionByConstant.h
        Functions/Inverse.c
        Functions/Inverse.h
        Functions/Mask.c
        Functions/Mask.h
        Functions/Mean.c
        Functions/Mean.h
        Functions/MultiplyByConstant.c
        Functions/MultiplyByConstant.h
        Functions/RemoveBias.c
        Functions/RemoveBias.h
        Functions/SequenceFunctionEdge.c
        Functions/SequenceFunctionEdge.h
        Functions/SquareRoot.c
        Functions/SquareRoot.h
        Functions/Switch.c
        Functions/Switch.h
        Functions/Transpose.c
        Functions/Transpose.h
        Functions/Variance.c
        Functions/Variance.h
        Parameters/RecurrentNeuralNetworkParameter.c
        Parameters/RecurrentNeuralNetworkParameter.h
        Parameters/TransformerParameter.c
        Parameters/TransformerParameter.h
        Performance/ClassificationPerformance.c
        Performance/ClassificationPerformance.h
        Sequence/LabelledSentence.c
        Sequence/LabelledSentence.h
        Sequence/SequenceCorpus.c
        Sequence/SequenceCorpus.h
        Sequence/LabelledVectorizedWord.c
        Sequence/LabelledVectorizedWord.h)

target_include_directories(SequenceProcessing
        PUBLIC
        ${SEQUENCE_PROCESSING_LOCAL_INCLUDE_DIRS})
target_link_libraries(SequenceProcessing
        PUBLIC
        sequence_processing_computational_graph
        sequence_processing_corpus
        sequence_processing_dictionary
        sequence_processing_math
        sequence_processing_util
        sequence_processing_data_structure
        sequence_processing_regular
        m)

function(sequence_processing_add_executable test_name test_source)
    add_executable(${test_name} ${test_source})
    target_include_directories(${test_name}
            PRIVATE
            ${SEQUENCE_PROCESSING_LOCAL_INCLUDE_DIRS})
    target_link_libraries(${test_name}
            PRIVATE
            SequenceProcessing)
endfunction()

function(sequence_processing_add_test test_name)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

sequence_processing_add_executable(SequenceCorpusTest ../test/SequenceCorpusTest.c)
sequence_processing_add_executable(ParameterSliceTest ../test/ParameterSliceTest.c)
sequence_processing_add_executable(FunctionSliceTest ../test/FunctionSliceTest.c)
sequence_processing_add_executable(RecurrentNeuralNetworkModelTest ../test/RecurrentNeuralNetworkModelTest.c)
sequence_processing_add_executable(GatedRecurrentUnitModelTest ../test/GatedRecurrentUnitModelTest.c)
sequence_processing_add_executable(LongShortTermMemoryModelTest ../test/LongShortTermMemoryModelTest.c)
sequence_processing_add_executable(TransformerTest ../test/TransformerTest.c)

sequence_processing_add_test(SequenceCorpusTest)
sequence_processing_add_test(ParameterSliceTest)
sequence_processing_add_test(FunctionSliceTest)
sequence_processing_add_test(RecurrentNeuralNetworkModelTest)

# The model-layer binaries below still compile, but they are not registered in
# ctest yet because their current runtime paths cross upstream sibling-library
# bugs outside SequenceProcessing-C:
# - GRU/LSTM hit ComputationalGraph-C concatenation execution corruption
# - Transformer currently also depends on unresolved Dictionary-C cleanup
#   behavior and the same staged graph-runtime boundary
