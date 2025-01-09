/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */

/*
The modifications made by hlwu primarily include the following:

1. Introduction of the assemble_A_M() Function: 
A new function, assemble_A_M(), has been added to facilitate 
the calculation of both the mass matrix and the Laplace matrix.

2. Removal of Adapted Mesh Functionality: 
The functionality related to adapted meshes has been removed 
to enhance simplicity.

3. Provision of an Interface (.prm File) : 
An interface in the form of a .prm file has been provided 
to simplify the parameter-setting process.
*/

// The program starts with the usual include files, all of which you should
// have seen before by now:
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/parsed_function.h>       // hlwu
#include <deal.II/base/parameter_acceptor.h>    // hlwu

#include <deal.II/grid/grid_in.h>

#include <cmath>
#include <string>

#include <fstream>
#include <iostream>

// Then the usual placing of all content of this program into a namespace and
// the importation of the deal.II namespace into the one we will work in:
namespace Step26
{
  using namespace dealii;

  // @sect3{The <code>DiffusionEquation</code> class}
  //
  // The next piece is the declaration of the main class of this program. It
  // follows the well trodden path of previous examples. If you have looked at
  // step-6, for example, the only thing worth noting here is that we need to
  // build two matrices (the mass and Laplace matrix) and keep the current and
  // previous time step's solution. We then also need to store the current
  // time, the size of the time step, and the number of the current time
  // step. The last of the member variables denotes the theta parameter
  // discussed in the introduction that allows us to treat the explicit and
  // implicit Euler methods as well as the Crank-Nicolson method and other
  // generalizations all in one program.
  //
  // As far as member functions are concerned, the only possible surprise is
  // that the <code>refine_mesh</code> function takes arguments for the
  // minimal and maximal mesh refinement level. The purpose of this is
  // discussed in the introduction.
  template <int dim>
  class DiffusionEquation : public ParameterAcceptor // hlwu
  {
  public:
    DiffusionEquation();
    void run();

  private:
    void setup_system();
    void solve_time_step();
    void output_results() const;
    void assemble_A_M();    // hlwu
    
    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> mass_matrix;     
    SparseMatrix<double> laplace_matrix; 
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;

    std::string   mesh_file;          // hlwu
    unsigned int  initial_global_refinement;

    double time = 0;
    double diff_lasting_time; // hlwu
    double time_step;
    unsigned int timestep_number = 0;
    const double theta;

    bool varied_diffusion_constant; // hlwu
    

    // hlwu
    ParameterAcceptorProxy<Functions::ParsedFunction<dim>>
      right_hand_side_function;
    ParameterAcceptorProxy<Functions::ParsedFunction<dim>>
      initial_value_function;
    ParameterAcceptorProxy<Functions::ParsedFunction<dim>>
      boundary_values_function;
    ParameterAcceptorProxy<Functions::ParsedFunction<dim>>
      diffusion_coefficient_function;
  };

  // @sect3{The <code>DiffusionEquation</code> implementation}
  //
  // It is time now for the implementation of the main class. Let's
  // start with the constructor which selects a linear element, a time
  // step constant at time_step and chooses the Crank Nicolson method
  // by setting $\theta=1/2$.
  template <int dim>
  DiffusionEquation<dim>::DiffusionEquation() : ParameterAcceptor("/Diffusion Equation/")
      , fe(1) // first-order Lagrange element
      , dof_handler(triangulation)
      , mesh_file("mesh/3d_new.msh")
      , initial_global_refinement(5)
      , diff_lasting_time(100)
      , time_step(1)
      , theta(0.5)
      , varied_diffusion_constant(true)
      , right_hand_side_function("/Diffusion Equation/Right hand side")
      , initial_value_function("/Diffusion Equation/Initial value")
      , boundary_values_function("/Diffusion Equation/Boundary values")
      , diffusion_coefficient_function("/Diffusion Equation/Diffusion coefficient")
  {
    add_parameter("Mesh file",
                  mesh_file,
                  "mesh used.");
    add_parameter("Initial global refinement",
                  initial_global_refinement,
                  "Number of times the mesh is refined globally before "
                  "starting the time stepping.");
    add_parameter("diffusion duration",
                  diff_lasting_time,
                  "Total duration in million years.");
    add_parameter("time step size",
                  time_step,
                  "Time step size in million years.");
    add_parameter("varied diffusion coefficient",
                  varied_diffusion_constant,
                  "Diffusion coefficient is a constant during diffsion.");
  }

  // @sect4{<code>DiffusionEquation::assemble_A_M</code>}
  //
  // The function assemble the mass matrix M and laplace matrix A
  // If the diffusion coefficient is an constant, then the matrix M only need to assemble once at t=0
  template <int dim>
  void DiffusionEquation<dim>::assemble_A_M()
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix_A(dofs_per_cell); // newly defined
    FullMatrix<double> cell_matrix_M(dofs_per_cell); // newly defined

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    diffusion_coefficient_function.set_time(time); // hlwu

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix_A = 0; 
      cell_matrix_M = 0;
      fe_values.reinit(cell);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        {
          for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
            {
              cell_matrix_A(i, j) += diffusion_coefficient_function.value(fe_values.quadrature_point(q_index)) * // hlwu: diffusion_coefficient_function(x_q)
                                     fe_values.shape_grad(i, q_index) *
                                     fe_values.shape_grad(j, q_index) *
                                     fe_values.JxW(q_index);
              // The M matrix is assembled only once, as the mesh remains unchanged, the M matrix is unchanged throughout
              if (timestep_number == 0)
              {
                cell_matrix_M(i, j) += fe_values.shape_value(i, q_index) *
                                     fe_values.shape_value(j, q_index) *
                                     fe_values.JxW(q_index);
              }             
            }
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
        {
          laplace_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix_A(i, j));
          if (timestep_number == 0) // The M matrix is assembled only once, as the mesh remains unchanged, the M matrix is unchanged throughout
          {
            mass_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix_M(i, j));
          }           
        }
      }
    }
  }

  // @sect4{<code>DiffusionEquation::setup_system</code>}
  //
  // The next function is the one that sets up the DoFHandler object,
  // computes the constraints, and sets the linear algebra objects
  // to their correct sizes. We also compute the mass and Laplace
  // matrix here by simply calling two functions in the library.
  //
  // Note that we do not take the hanging node constraints into account when
  // assembling the matrices (both functions have an AffineConstraints argument
  // that defaults to an empty object). This is because we are going to
  // condense the constraints in run() after combining the matrices for the
  // current time-step.
  template <int dim>
  void DiffusionEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    // assembling the mass matrix M and laplace matrix A
    assemble_A_M();

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  // @sect4{<code>DiffusionEquation::solve_time_step</code>}
  //
  // The next function is the one that solves the actual linear system
  // for a single time step. There is nothing surprising here:
  template <int dim>
  void DiffusionEquation<dim>::solve_time_step()
  {
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step() << " CG iterations."
              << std::endl;
  }

  // @sect4{<code>DiffusionEquation::output_results</code>}
  //
  // Neither is there anything new in generating graphical output other than the
  // fact that we tell the DataOut object what the current time and time step
  // number is, so that this can be written into the output file:
  template <int dim>
  void DiffusionEquation<dim>::output_results() const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");

    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
        std::to_string(time) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
  }

  // @sect4{<code>DiffusionEquation::run</code>}
  //
  // This is the main driver of the program, where we loop over all
  // time steps. At the top of the function, we refine the mesh. 
  // Then we create a mesh, initialize the various objects we will
  // work with, and interpolate the initial solution onto
  // out mesh.
  template <int dim>
  void DiffusionEquation<dim>::run()
  {
    // hlwu reading the mesh
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f(mesh_file);
    std::cout << "mesh file: " << mesh_file << std::endl;
    gridin.read_msh(f);

    triangulation.refine_global(initial_global_refinement);

    setup_system();

    Vector<double> tmp;
    Vector<double> forcing_terms;

    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());

    VectorTools::interpolate(dof_handler,
                             initial_value_function,
                             old_solution);
    solution = old_solution;
    output_results(); // output the initial result

    // Then we start the main loop until the computed time exceeds our
    // end time. The first task is to build the right hand
    // side of the linear system we need to solve in each time step.
    // Recall that it contains the term $MU^{n-1}-(1-\theta)k_n AU^{n-1}$.
    // We put these terms into the variable system_rhs, with the
    // help of a temporary vector:
    while (time <= diff_lasting_time)
    {
      time += time_step;
      ++timestep_number;

      std::cout << "Time step " << timestep_number << " at t=" << time
                << std::endl;

      mass_matrix.vmult(system_rhs, old_solution);

      laplace_matrix.vmult(tmp, old_solution);
      system_rhs.add(-(1 - theta) * time_step, tmp);

      // The second piece is to compute the contributions of the source
      // terms. This corresponds to the term $k_n
      // \left[ (1-\theta)F^{n-1} + \theta F^n \right]$. The following
      // code calls VectorTools::create_right_hand_side to compute the
      // vectors $F$, where we set the time of the right hand side
      // (source) function before we evaluate it. The result of this
      // all ends up in the forcing_terms variable:
      right_hand_side_function.set_time(time);
      VectorTools::create_right_hand_side(dof_handler,
                                          QGauss<dim>(fe.degree + 1),
                                          right_hand_side_function,
                                          tmp);
      forcing_terms = tmp;
      forcing_terms *= time_step * theta;

      right_hand_side_function.set_time(time - time_step);
      VectorTools::create_right_hand_side(dof_handler,
                                          QGauss<dim>(fe.degree + 1),
                                          right_hand_side_function,
                                          tmp);

      forcing_terms.add(time_step * (1 - theta), tmp);

      // Next, we add the forcing terms to the ones that
      // come from the time stepping, and also build the matrix
      // $M+k_n\theta A$ that we have to invert in each time step.
      // The final piece of these operations is to eliminate
      // hanging node constrained degrees of freedom from the
      // linear system:
      system_rhs += forcing_terms;

      system_matrix.copy_from(mass_matrix);
      system_matrix.add(theta * time_step, laplace_matrix);

      constraints.condense(system_matrix, system_rhs);

      // There is one more operation we need to do before we
      // can solve it: boundary values. To this end, we create
      // a boundary value object, set the proper time to the one
      // of the current time step, and evaluate it as we have
      // done many times before. The result is used to also
      // set the correct boundary values in the linear system:
      {
        boundary_values_function.set_time(time);

        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 boundary_values_function,
                                                 boundary_values);

        MatrixTools::apply_boundary_values(boundary_values,
                                           system_matrix,
                                           solution,
                                           system_rhs);
      }

      // With this out of the way, all we have to do is solve the
      // system, generate graphical data, and...
      solve_time_step();
      
      // hlwu: if the one to the last step, change the time step in order to 
      // get to the diff_lasting_time
      if (fabs(diff_lasting_time - time) <= 1e-10) 
      {
        output_results(); 
        break;  
      }
      else if (time < diff_lasting_time &&  
                fabs(diff_lasting_time - time) < time_step)
      {
        time_step = fabs(diff_lasting_time - time); 
      }
      
      // The time loop and, indeed, the main part of the program ends
      // with starting into the next time step by setting old_solution
      // to the solution we have just computed.

      // hlwu
      // 对容器进行清空,进行下一时间步计算
      tmp.reinit(solution.size());
      forcing_terms.reinit(solution.size());
      system_rhs.reinit(solution.size());
      system_matrix.reinit(sparsity_pattern);  

      // 重新组装A矩阵, 当扩散系数随时间与空间发生变化
      if (varied_diffusion_constant)
      {
        laplace_matrix.reinit(sparsity_pattern);
        assemble_A_M();
      }
      
      old_solution = solution;     
    }
  }
} // namespace Step26

// @sect3{The <code>main</code> function}
//
// Having made it this far,  there is, again, nothing
// much to discuss for the main function of this
// program: it looks like all such functions since step-6.
int main(int argc, char **argv)
{
  try
  {
    using namespace Step26;
    DiffusionEquation<2> diffusion_equation_solver;
    const std::string input_filename =  
        (argc > 1 ? argv[1] : "diffusion_equation.prm"); // hlwu
    ParameterAcceptor::initialize(input_filename, "diffusion_equation_used.prm"); // hlwu
    diffusion_equation_solver.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}