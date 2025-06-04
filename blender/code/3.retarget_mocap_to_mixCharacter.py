import os
import bpy
import rokoko


# for running once
CHARACTER_FILE_PATH = "../data/characters-mixamo-processed/Abe-T-Pose.blend"
ACTION_FILE_PATH = "../data/action-mocap/proud_01_000.bvh"
OUTPUT_FILE_PATH = "../output/3.preview_render_animate_with_mocap_proud_01_000.mp4"

# for running with env var
# CHARACTER_FILE_PATH = os.environ.get("CHARACTER", "../data/characters-processed/Abe-T-Pose.blend")
# ACTION_FILE_PATH = os.environ.get("ACTION", "../data/action-mocap/proud_01_000.bvh")
# OUTPUT_FILE_PATH = os.environ.get("OUTPUT", "../output/3.preview_render_animate_with_mocap_proud_01_000.mp4")


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


def import_bvh(filepath, update_scene_fps=False):
    bpy.ops.import_anim.bvh(
        filepath=filepath,
        update_scene_fps=update_scene_fps,
    )
    return bpy.context.selected_objects



clear_all_objects()



with bpy.data.libraries.load(CHARACTER_FILE_PATH, link=False) as (data_from, data_to):
    print("Objects in file:", data_from.objects)
    # Choose what to load
    data_to.objects = data_from.objects

# Link objects into scene
for obj in data_to.objects:
    if obj is not None:
        bpy.context.collection.objects.link(obj)
        print(f"✅ Imported: {obj.name}")



action_objects = import_bvh(ACTION_FILE_PATH, True)

frame_range = action_objects[0].animation_data.action.frame_range
bpy.context.scene.frame_start = int(frame_range[0])
bpy.context.scene.frame_end = int(frame_range[1])




rokoko.bpy.context.scene.rsl_retargeting_armature_source = action_objects[0]
charectar_object = bpy.data.objects.get("Armature")
rokoko.bpy.context.scene.rsl_retargeting_armature_target = charectar_object
rokoko.bpy.ops.rsl.build_bone_list()
rokoko.bpy.ops.rsl.retarget_animation()



def setup_camera():
    bpy.ops.object.camera_add(location=(0, -6, 3), rotation=(1.2, 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

setup_camera()

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


