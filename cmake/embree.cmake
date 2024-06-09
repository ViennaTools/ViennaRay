macro(setup_embree_env TARGET OUTPUT)
  message(STATUS "[ViennaRay] Setting up Embree-Environment for ${TARGET}")

  if(NOT TARGET embree)
    message(WARNING "[ViennaRay] Could not find Embree-Target")
    return()
  endif()

  add_custom_command(
    TARGET ${TARGET}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:embree> ${OUTPUT})
endmacro()
