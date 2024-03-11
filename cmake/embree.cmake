macro(setup_embree_env TARGET OUTPUT)
    message(STATUS "[ViennaRay] Setting up Embree-Environment for ${TARGET}")

    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:tbb>
              ${OUTPUT})

    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:embree>
              ${OUTPUT})
endmacro()
