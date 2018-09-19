import os
import matplotlib.pyplot as plt
from inversegraphics_generator.obj_generator import ObjGenerator
from inversegraphics_generator.iqtest_objs import get_data_dir

OUT_PATH = os.path.join(get_data_dir(), "object-{:06d}.obj")

# the size of the snake is randomly sampled
# from this interval
SNEK_LEN_MIN = 5
SNEK_LEN_MAX = 10

GRID_SIZE = 4  # the grid for the snake is N x N x N

CUBE_SIZE = 1.0  # cubes are unit-sized

### IMPORTANT
OBJECTS_TO_GENERATE = 1000

if __name__ == '__main__':
    og = ObjGenerator(GRID_SIZE, CUBE_SIZE)
    ## test with one cube
    # v, f = make_cube(CUBE_SIZE)
    # plot_cube(v, f)

    # test with one grid
    g = og.walk_snek(SNEK_LEN_MIN, SNEK_LEN_MAX)
    print(g)

    v, f = og.grid_to_cubes(g)
    og.write_obj(g,v,f,0,OUT_PATH)
    og.plot_cube(v, f)

    # v, f = og.grid_to_cubes(np.flip(g,axis=0))
    # og.plot_cube(v, f)
    #
    # v, f = og.grid_to_cubes(np.flip(g,axis=1))
    # og.plot_cube(v, f)
    #
    # v, f = og.grid_to_cubes(np.flip(g,axis=2))
    # og.plot_cube(v, f)

    plt.show()

    # for obj_idx in tqdm(range(OBJECTS_TO_GENERATE)):
    #     g = walk_snek()
    #     v, f = grid_to_cubes(g)
    #     write_obj(g, v, f, obj_idx)
    #     # plot_cube(v, f)
