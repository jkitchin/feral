// FeralSolverInterface implementation — see header for design notes.

#include "IpFeralSolverInterface.hpp"

namespace Ipopt
{

static ESymSolverStatus to_ipopt_status(int s)
{
   switch (s) {
      case FERAL_SUCCESS:        return SYMSOLVER_SUCCESS;
      case FERAL_SINGULAR:       return SYMSOLVER_SINGULAR;
      case FERAL_WRONG_INERTIA:  return SYMSOLVER_WRONG_INERTIA;
      case FERAL_FATAL:
      default:                   return SYMSOLVER_FATAL_ERROR;
   }
}

FeralSolverInterface::FeralSolverInterface()
   : solver_(0),
     dim_(0),
     nonzeros_(0),
     have_structure_(false),
     have_factor_(false),
     negevals_(0)
{
   solver_ = feral_new();
}

FeralSolverInterface::~FeralSolverInterface()
{
   if (solver_) {
      feral_free(solver_);
      solver_ = 0;
   }
}

bool FeralSolverInterface::InitializeImpl(
   const OptionsList& /*options*/,
   const std::string& /*prefix*/
)
{
   // POC: no options to read. Solver::new() defaults are used
   // throughout. Returning true tells Ipopt we initialized
   // successfully.
   return solver_ != 0;
}

ESymSolverStatus FeralSolverInterface::InitializeStructure(
   Index        dim,
   Index        nonzeros,
   const Index* ia,
   const Index* ja
)
{
   if (!solver_) return SYMSOLVER_FATAL_ERROR;

   dim_ = dim;
   nonzeros_ = nonzeros;
   have_structure_ = false;
   have_factor_ = false;

   // Ipopt's Index is `int` by default in 3.14; feral_set_structure
   // takes int*. If Ipopt was built with --with-intsize=64 we'd
   // need to copy/cast; for the POC we assume the default int.
   int status = feral_set_structure(
      solver_,
      static_cast<int>(dim),
      static_cast<int>(nonzeros),
      ia,
      ja);

   if (status == FERAL_SUCCESS) {
      have_structure_ = true;
      return SYMSOLVER_SUCCESS;
   }
   return to_ipopt_status(status);
}

Number* FeralSolverInterface::GetValuesArrayPtr()
{
   if (!solver_ || !have_structure_) return 0;
   return feral_values_ptr(solver_);
}

ESymSolverStatus FeralSolverInterface::MultiSolve(
   bool         new_matrix,
   const Index* /*ia*/,
   const Index* /*ja*/,
   Index        nrhs,
   Number*      rhs_vals,
   bool         check_NegEVals,
   Index        numberOfNegEVals
)
{
   if (!solver_ || !have_structure_) return SYMSOLVER_FATAL_ERROR;

   if (new_matrix || !have_factor_) {
      int s = feral_factor(
         solver_,
         check_NegEVals ? 1 : 0,
         static_cast<int>(numberOfNegEVals));
      if (s != FERAL_SUCCESS) {
         // WRONG_INERTIA still leaves the factor stored — feral
         // returns the actual negative-eval count via
         // feral_num_neg below, and Ipopt may choose to solve
         // against it anyway. SINGULAR / FATAL skip the solve.
         if (s == FERAL_WRONG_INERTIA) {
            negevals_ = feral_num_neg(solver_);
            have_factor_ = true;
         }
         return to_ipopt_status(s);
      }
      negevals_ = feral_num_neg(solver_);
      have_factor_ = true;
   }

   int s = feral_solve(
      solver_,
      static_cast<int>(nrhs),
      rhs_vals);
   return to_ipopt_status(s);
}

Index FeralSolverInterface::NumberOfNegEVals() const
{
   return negevals_;
}

bool FeralSolverInterface::IncreaseQuality()
{
   // POC: no escalation. Ipopt falls back to PDPerturbationHandler
   // for handling WRONG_INERTIA / SINGULAR.
   return false;
}

} // namespace Ipopt
