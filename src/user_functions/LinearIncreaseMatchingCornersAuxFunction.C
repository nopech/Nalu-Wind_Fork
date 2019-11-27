/* 
 * File:   HigherOrderTri2DSCS.C
 * Author: Raphael Lindegger
 * 
 * Created on November 27, 2019, 01:11 PM
 */


#include <user_functions/LinearIncreaseMatchingCornersAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

LinearIncreaseMatchingCornersAuxFunction::LinearIncreaseMatchingCornersAuxFunction() :
  AuxFunction(0, 1)
{
  // Nothing to do
}

void
LinearIncreaseMatchingCornersAuxFunction::do_evaluate(
  const double * coords,
  const double /*time*/,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for(unsigned p=0; p < numPoints; ++p) {
    
    if (spatialDimension == 2) {
      const int offset = 2*p;
      const double cX = coords[offset + 0];
      const double cY = coords[offset + 1];

      fieldPtr[p] = cX + cY + cX*cY;
    }
    else if (spatialDimension == 3) {
      const int offset = 3*p;
      const double cX = coords[offset + 0];
      const double cY = coords[offset + 1];
      const double cZ = coords[offset + 2];

      fieldPtr[0] = cX + cY + cZ + cX*cY*cZ;
    }
  }
}

} // namespace nalu
} // namespace Sierra
