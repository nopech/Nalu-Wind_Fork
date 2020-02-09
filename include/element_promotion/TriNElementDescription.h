/* 
 * File:   TriNElementDescription.h
 * Author: Raphael Lindegger
 *
 * Created on November 2, 2019, 2:02 PM
 */

#ifndef TriNElementDescription_h
#define TriNElementDescription_h

#include <stddef.h>
#include <map>
#include <memory>
#include <vector>
#include <element_promotion/ElementDescription.h>

namespace sierra {
namespace nalu {

struct TriNElementDescription final: public ElementDescription
{
public:
  TriNElementDescription(std::vector<double> nodeLocs);
private:
  void set_subelement_connectivity();
  std::vector<ordinal_type> edge_node_ordinals();
  void set_edge_node_connectivities();
  std::vector<ordinal_type> volume_node_ordinals();
  void set_volume_node_connectivities();
  void set_subelement_connectivites();
  void set_side_node_ordinals();
  std::pair<ordinal_type,ordinal_type> get_edge_offsets(ordinal_type i, ordinal_type j, ordinal_type edge_offset);
  void set_base_node_maps();
  void set_tensor_product_node_mappings();
  void set_boundary_node_mappings();
  void set_isoparametric_coordinates();
  ordinal_type& nmap(ordinal_type i, ordinal_type j ) { return nodeMap.at(i+nodes1D*j); };
  std::vector<ordinal_type>& inmap(ordinal_type j) { return inverseNodeMap.at(j); };
  std::vector<double> scaleNodeLocs(std::vector<double> in_nodeLocs);
};

} // namespace nalu
} // namespace Sierra

#endif
