import coord_descent
import incremental_coord_descent
import parallel_coord_descent
import coord_descent_with_moves

alg_name_map = dict(
    coord_descent=coord_descent,
    incremental_coord_descent=incremental_coord_descent,
    parallel_coord_descent=parallel_coord_descent,
    coord_descent_with_moves=coord_descent_with_moves)