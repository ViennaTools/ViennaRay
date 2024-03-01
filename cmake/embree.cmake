macro(setup_embree_env TARGET OUTPUT)
  if(VIENNARAY_DISABLE_COPY OR NOT MSVC OR VIENNARAY_SYSTEM_EMBREE)
    message(STATUS "[ViennaRay] Skipping Embree-Environment setup for ${TARGET}")
  else()
    message(STATUS "[ViennaRay] Setting up Embree-Environment for ${TARGET}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/${OUTPUT}>)

    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:tbb>
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:embree>
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif()
endmacro()
