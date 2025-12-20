# In CMake, functions have their own scope, whereas macros use the scope of the caller.
function(add_gpu_deps target_name)

  add_dependencies(${target_name} disk_pipeline triangle_pipeline line_pipeline callable_wrapper
                   norm_kernels)
endfunction()
