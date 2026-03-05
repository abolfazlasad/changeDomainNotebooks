import bpy
import mixamo_rig

CHARACTER_FILE_PATH = "../data/characters-mixamo/Abe-T-Pose.fbx"
ACTION_FILE_PATH = "../data/action-mixamo/Situps.fbx"
OUTPUT_FILE_PATH = "../output/1.preview_render_mixamo_char_mixamo_anim.mp4"

def clear_all_objects():
    # Select all objects
    bpy.ops.object.select_all(action='SELECT')
    # Delete selected objects
    bpy.ops.object.delete(use_global=False)

def delete_object(objects):
    # Deselect all first
    bpy.ops.object.select_all(action='DESELECT')
    # Select and delete each object in objects
    for obj in objects:
        obj.select_set(True)
    # Delete
    bpy.ops.object.delete()

def import_fbx(filepath):
    bpy.ops.import_scene.fbx(filepath=filepath)
    return bpy.context.selected_objects

def made_animate(charecter_objects, action_objects):
    bpy.context.view_layer.objects.active = charecter_objects[0]
    mixamo_rig.bpy.ops.mr.make_rig()
    mixamo_rig.bpy.context.scene.mix_source_armature = action_objects[0]
    mixamo_rig.bpy.ops.mr.import_anim_to_rig()





clear_all_objects()


charecter_objects = import_fbx(CHARACTER_FILE_PATH)
action_objects = import_fbx(ACTION_FILE_PATH)

made_animate(charecter_objects, action_objects)
delete_object(action_objects)




def setup_camera(target_obj):
    import math

    # Add a camera
    bpy.ops.object.camera_add(location=(0, -5, 2), rotation=(math.radians(75), 0, 0))
    camera = bpy.context.object

    # Set camera as active
    bpy.context.scene.camera = camera

    # Add empty to use as tracking target (optional for flexibility)
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=target_obj.location)
    empty = bpy.context.object

    # Parent empty to target so it follows it
    empty.parent = target_obj

    # Add 'Track To' constraint to camera
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    return camera


def setup_camera():
    bpy.ops.object.camera_add(location=(0, -5, 2), rotation=(1.2, 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

setup_camera()


# Set frame range
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 60




# Set low-quality render settings
scene = bpy.context.scene
render = scene.render

render.engine = 'BLENDER_EEVEE'  # Use EEVEE for fast rendering (vs CYCLES)

render.resolution_x = 1280
render.resolution_y = 720
render.resolution_percentage = 100  # Half resolution

scene.render.film_transparent = False
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'

# Output path (change this to your preferred directory)
render.filepath = OUTPUT_FILE_PATH

# Run the render
bpy.ops.render.render(animation=True)


