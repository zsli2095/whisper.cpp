if (WHISPER_SDL2)
    set(TARGET whisper-stream2)
    add_executable(${TARGET} stream2.cpp)

    include(DefaultTargetOptions)
    include_directories("/opt/homebrew/include")
    #link_directories("/opt/homebrew/lib")
    #find_library(ZMQ_LIBRARIES NAMES zmq)
    target_link_libraries(${TARGET} PRIVATE "/opt/homebrew/lib/libzmq.dylib" common common-sdl whisper ${CMAKE_THREAD_LIBS_INIT})

    install(TARGETS ${TARGET} RUNTIME)
endif ()
