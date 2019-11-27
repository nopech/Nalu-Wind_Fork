/* 
 * File:   HigherOrderTri2DSCS.C
 * Author: Raphael Lindegger
 * 
 * Created on November 27, 2019, 01:11 PM
 */

#ifndef LinearIncreaseMatchingCornersAuxFunction_h
#define LinearIncreaseMatchingCornersAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

/** Linear increasing value depending on coordinates
 *
 *  This function is used as a dirichlet boundary condition for diffusion validation 
 *  cases.
 *  The function has the property that the boundary condition value does linearly change
 *  and does not cause a jump at the corner of a square. It does also represent the exact
 *  solution.
 */
class LinearIncreaseMatchingCornersAuxFunction : public AuxFunction
{
public:

  LinearIncreaseMatchingCornersAuxFunction();

  virtual ~LinearIncreaseMatchingCornersAuxFunction() {}
  
  using AuxFunction::do_evaluate;
  virtual void do_evaluate(
    const double * coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double * fieldPtr,
    const unsigned fieldSize,
    const unsigned beginPos,
    const unsigned endPos) const;
};

} // namespace nalu
} // namespace Sierra

#endif
