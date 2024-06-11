macro(setup_tbb_env TARGET OUTPUT)
  message(STATUS "[ViennaRay] Setting up TBB-Environment for ${TARGET}")

  if(NOT TARGET tbb)
    message(WARNING "[ViennaRay] Could not find TBB-Target")
    return()
  endif()

  add_custom_command(
    TARGET ${TARGET}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:tbb> ${OUTPUT})
endmacro()
