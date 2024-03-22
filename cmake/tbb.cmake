macro(setup_tbb_env TARGET OUTPUT)
  message(STATUS "[ViennaRay] Setting up TBB-Environment for ${TARGET}")

  add_custom_command(
    TARGET ${TARGET}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:tbb> ${OUTPUT})
endmacro()
