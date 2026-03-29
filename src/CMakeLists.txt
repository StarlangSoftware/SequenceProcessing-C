add_library(SequenceProcessing
        SequenceProcessing.c
        SequenceProcessing.h
        Sequence/LabelledSentence.c
        Sequence/LabelledSentence.h
        Sequence/SequenceCorpus.c
        Sequence/SequenceCorpus.h
        Sequence/LabelledVectorizedWord.c
        Sequence/LabelledVectorizedWord.h)

target_include_directories(SequenceProcessing
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE
        ${SEQUENCE_PROCESSING_C_LOCAL_CORPUS_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src/Dictionary
        ${SEQUENCE_PROCESSING_C_LOCAL_MATH_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_UTIL_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src)

find_package(util_c QUIET)
find_package(math_c QUIET)
find_package(dictionary_c QUIET)
find_package(corpus_c QUIET)
find_package(data_structure_c QUIET)
find_package(regular_c QUIET)

target_link_libraries(SequenceProcessing
        PRIVATE
        $<$<TARGET_EXISTS:util_c::util_c>:util_c::util_c>
        $<$<TARGET_EXISTS:math_c::math_c>:math_c::math_c>
        $<$<TARGET_EXISTS:dictionary_c::dictionary_c>:dictionary_c::dictionary_c>
        $<$<TARGET_EXISTS:corpus_c::corpus_c>:corpus_c::corpus_c>
        $<$<TARGET_EXISTS:data_structure_c::data_structure_c>:data_structure_c::data_structure_c>
        $<$<TARGET_EXISTS:regular_c::regular_c>:regular_c::regular_c>)

add_executable(SequenceCorpusTest
        ../test/SequenceCorpusTest.c)

target_include_directories(SequenceCorpusTest
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SEQUENCE_PROCESSING_C_LOCAL_CORPUS_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DICTIONARY_ROOT}/src/Dictionary
        ${SEQUENCE_PROCESSING_C_LOCAL_MATH_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_UTIL_ROOT}/src
        ${SEQUENCE_PROCESSING_C_LOCAL_DATA_STRUCTURE_ROOT}/src)

target_link_libraries(SequenceCorpusTest
        PRIVATE
        SequenceProcessing
        $<$<TARGET_EXISTS:util_c::util_c>:util_c::util_c>
        $<$<TARGET_EXISTS:math_c::math_c>:math_c::math_c>
        $<$<TARGET_EXISTS:dictionary_c::dictionary_c>:dictionary_c::dictionary_c>
        $<$<TARGET_EXISTS:corpus_c::corpus_c>:corpus_c::corpus_c>
        $<$<TARGET_EXISTS:data_structure_c::data_structure_c>:data_structure_c::data_structure_c>
        $<$<TARGET_EXISTS:regular_c::regular_c>:regular_c::regular_c>)
