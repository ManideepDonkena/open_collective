"""Model zoo: alignment-only, cohesive, grouping, formation, consensus."""
from .alignment import (VicsekModel, PerceptionQuantum, SlowFastPerception,
                        KuramotoModel)
from .cohesive import (BoidsModel, CouzinModel, DOrsognaModel,
                       CuckerSmaleModel, OlfatiSaberModel)
from .grouping import MultiGroupFlock, assign_groups, init_grouped_state
from .formation import (DisplacementFormation, DistanceFormation, LeaderFollower,
                        CyclicPursuit, regular_polygon, grid_shape, v_shape,
                        complete_graph, cycle_graph, distance_matrix_from_shape,
                        algebraic_connectivity, is_infinitesimally_rigid)
from .consensus import (DeGroot, FriedkinJohnsen, SignedFJ, AltafiniBipartite,
                        GroupConsensus, structural_balance, signed_laplacian,
                        condensation_leaders, is_primitive)

ALIGNMENT_ONLY = [VicsekModel, PerceptionQuantum, SlowFastPerception, KuramotoModel]
COHESIVE = [BoidsModel, CouzinModel, DOrsognaModel, CuckerSmaleModel, OlfatiSaberModel]
GROUPING = [MultiGroupFlock]
FORMATION = [DisplacementFormation, DistanceFormation, LeaderFollower, CyclicPursuit]
CONSENSUS = [DeGroot, FriedkinJohnsen, SignedFJ, AltafiniBipartite, GroupConsensus]
