import bpy
import mixamo_rig

import math

CHARACTER_FILE_PATH = "../data/characters-mixamo/Mousey-T-Pose.fbx"

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


def import_bvh(filepath):
    bpy.ops.import_anim.bvh(
        filepath=filepath,
    )
    return bpy.context.selected_objects


bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 60


clear_all_objects()

charecter_objects = import_fbx(CHARACTER_FILE_PATH)

# Select and make it the active object
bpy.ops.object.select_all(action='DESELECT')
charecter_objects[0].select_set(True)
bpy.context.view_layer.objects.active = charecter_objects[0]

# Switch to OBJECT mode first (required)
bpy.ops.object.mode_set(mode='OBJECT')

# Then switch to POSE mode
bpy.ops.object.mode_set(mode='POSE')

bpy.ops.pose.transforms_clear()



##############################  change reset pose ##############################

armature = charecter_objects[0]

# Deselect all pose bones first
for b in armature.data.bones:
    b.select = False
# Select the target bone
armature.data.bones["mixamorig:RightUpLeg"].select = True
armature.data.bones.active = armature.data.bones["mixamorig:RightUpLeg"]
# Get pose bone to rotate
pose_bone = armature.pose.bones["mixamorig:RightUpLeg"]
pose_bone.rotation_mode = 'XYZ'
# Rotate 10 degrees on Y axis
pose_bone.rotation_euler[2] += math.radians(-11)


# Deselect all pose bones first
for b in armature.data.bones:
    b.select = False
# Select the target bone
armature.data.bones["mixamorig:LeftUpLeg"].select = True
armature.data.bones.active = armature.data.bones["mixamorig:LeftUpLeg"]
# Get pose bone to rotate
pose_bone = armature.pose.bones["mixamorig:LeftUpLeg"]
pose_bone.rotation_mode = 'XYZ'
# Rotate 10 degrees on Y axis
pose_bone.rotation_euler[2] += math.radians(11)








bpy.ops.object.mode_set(mode='OBJECT')

# Deselect all first
bpy.ops.object.select_all(action='DESELECT')

# Loop through all objects and select only meshes
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        break
