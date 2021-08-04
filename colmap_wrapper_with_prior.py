import os
import subprocess

colmap = 'colmap'

# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# Two-stage COLMAP (feature extraction/matching first and then triangulator)
# Fix image ordering
# Due to COLMAP processing the images in parallel, the order is different sometimes.
def run_feature(basedir, match_type):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    feature_extractor_args = [
        colmap, 'feature_extractor', 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--image_path', os.path.join(basedir, 'images'),
            '--ImageReader.camera_model', 'SIMPLE_PINHOLE',
            '--SiftExtraction.estimate_affine_shape', '1',
            '--SiftExtraction.domain_size_pooling', '1',
    ]

    feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
    logfile.write(feat_output)
    print('Features extracted')

    exhaustive_matcher_args = [
        colmap, match_type, 
            '--database_path', os.path.join(basedir, 'database.db'),
            '--SiftMatching.guided_matching', '1'
    ]

    match_output = ( subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
    logfile.write(match_output)
    print('Features matched')
    
    logfile.close()

def run_triangulate(basedir):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'a')
    
    triangulator_args = [
        colmap, 'point_triangulator',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'images'),
            '--input_path', os.path.join(basedir, 'sparse', '0'),
            '--output_path', os.path.join(basedir, 'sparse', '0'),
    ]

    map_output = ( subprocess.check_output(triangulator_args, universal_newlines=True) )
    logfile.write(map_output)
    print('Sparse map created')

    logfile.close()
    
    # print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )

def run_bundle_adjust(basedir):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'a')

    os.makedirs(os.path.join(basedir, 'refined', '0'))
    
    bundle_adjuster_args = [
        colmap, 'bundle_adjuster',
            '--input_path', os.path.join(basedir, 'sparse', '0'),
            '--output_path', os.path.join(basedir, 'refined', '0'),
    ]

    bundle_output = ( subprocess.check_output(bundle_adjuster_args, universal_newlines=True) )
    logfile.write(bundle_output)
    print('Bundle adjustment completed')

    logfile.close()
    
    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )