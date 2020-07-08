cmake_minimum_required(VERSION 3.11)
project(bader Fortran)

set(SRC
    bader_mod.f90
    charge_mod.f90
    chgcar_mod.f90
    critpoint_mod.f90
    cube_mod.f90
    dsyevj3_mod.f
    io_mod.f90
    ions_mod.f90
    kind_mod.f90
    linsolve_mod.f90
    main.f90
    matrix_mod.f90
    multipole_mod.f90
    options_mod.f90
    results_mod.f90
    voronoi_mod.f90
    wavecar_mod.f90
    weight_mod.f90
    )

add_executable(bader ${SRC})
