# Script taken from doing the needed operation
# (Filters > Remeshing, Simplification and Reconstruction >
# Quadric Edge Collapse Decimation, with parameters:
# 0.9 percentage reduction (10%), 0.3 Quality threshold (70%)
# Target number of faces is ignored with those parameters
# conserving face normals, planar simplification and
# post-simplimfication cleaning)
# And going to Filter > Show current filter script

filter_script_mlx = """
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Remove Zero Area Faces"/>
<filter name="Remove Unreferenced Vertices"/>
<filter name="Remove Duplicate Faces"/>
<filter name="Remove Duplicate Vertices"/>

 <filter name="Transform: Scale, Normalize">
  <Param description="X Axis" value="1" name="axisX" type="RichFloat"/>
  <Param description="Y Axis" value="1" name="axisY" type="RichFloat"/>
  <Param description="Z Axis" value="1" name="axisZ" type="RichFloat"/>
  <Param description="Uniform Scaling" value="true" name="uniformFlag" type="RichBool"/>
  <Param enum_val2="custom point" enum_cardinality="3" description="Center of scaling:" enum_val1="barycenter" value="2" name="scaleCenter" enum_val0="origin" type="RichEnum"/>
  <Param x="0" y="0" description="Custom center" z="0" name="customCenter" type="RichPoint3f"/>
  <Param description="Scale to Unit bbox" value="true" name="unitFlag" type="RichBool"/>
  <Param description="Freeze Matrix" value="true" name="Freeze" type="RichBool"/>
  <Param description="Apply to all visible Layers" value="false" name="allLayers" type="RichBool"/>
 </filter>
</FilterScript>

"""



# <filter name="Simplification: Quadric Edge Collapse Decimation">
#   <Param  description="Target number of faces" value="3000" name="TargetFaceNum" type="RichInt"/>
#   <Param  description="Percentage reduction (0..1)" value="0" name="TargetPerc" type="RichFloat"/>
#   <Param  description="Quality threshold" value="0.3" name="QualityThr" type="RichFloat"/>
#   <Param  description="Preserve Boundary of the mesh" value="false" name="PreserveBoundary" type="RichBool"/>
#   <Param  description="Boundary Preserving Weight" value="1" name="BoundaryWeight" type="RichFloat"/>
#   <Param  description="Preserve Normal" value="true" name="PreserveNormal" type="RichBool"/>
#   <Param  description="Preserve Topology" value="false" name="PreserveTopology" type="RichBool"/>
#   <Param  description="Optimal position of simplified vertices" value="true" name="OptimalPlacement" type="RichBool"/>
#   <Param  description="Planar Simplification" value="true" name="PlanarQuadric" type="RichBool"/>
#   <Param tooltip="Use the Per-Vertex quality as a weighting factor for the simplification. The weight is used as a error amplification value, so a vertex with a high quality value will not be simplified and a portion of the mesh with low quality values will be aggressively simplified." description="Weighted Simplification" value="false" name="QualityWeight" type="RichBool"/>
#   <Param  description="Post-simplification cleaning" value="true" name="AutoClean" type="RichBool"/>
#   <Param description="Simplify only selected faces" value="false" name="Selected" type="RichBool"/>
#  </filter>




#  <filter name="Close Holes">
#   <Param name="MaxHoleSize" description="Max size to be closed " type="RichInt" value="30"/>
#   <Param name="Selected" description="Close holes with selected faces" type="RichBool" value="false"/>
#   <Param name="NewFaceSelected" description="Select the newly created faces" type="RichBool" value="false"/>
#   <Param name="SelfIntersection" description="Prevent creation of selfIntersecting faces" type="RichBool" value="true"/>
#  </filter>
#


#  <filter name="Remove Zero Area Faces"/>
# <filter name="Remove Unreferenced Vertices"/>
# <filter name="Remove T-Vertices by Edge Collapse">
#  <Param description="Ratio" type="RichFloat" name="Threshold" value="40"/>
#  <Param description="Iterate until convergence" type="RichBool"   name="Repeat" value="true"/>
# </filter>
# <filter name="Remove Duplicate Faces"/>
# <filter name="Remove Duplicate Vertices"/>
# <filter name="Snap Mismatched Borders">
#  <Param description="Edge Distance Ratio" type="RichFloat" name="EdgeDistRatio" value="0.01"/>
#  <Param description="UnifyVertices" type="RichBool" name="UnifyVertices" value="true"/>
# </filter>