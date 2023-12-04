using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Xml;
using System.IO;
using System;
using UnityEditor;

/* Written by Kexin ZHENG 
* Load FBX animations from file with separate frame informations
*/
public static class FBXAnimationLoader{
    public static MotionAnimation LoadFBXAnimation(string name, string path){
        MotionAnimation curr_animation = new MotionAnimation(name);
        GameObject fbx_obj = Resources.Load<GameObject>("AnimationClip/Jump");
        AnimationClip motion_clip = Resources.Load<AnimationClip>(path);
        int total_frame = (int)(motion_clip.length * 30.0f); // total number of frame
        float delta_t = 1.0f/30.0f;
        for(int i = 0; i < total_frame;i++){
            var curr_t = i * delta_t;
            motion_clip.SampleAnimation(fbx_obj, curr_t);
            MotionFrame curr_frame = new MotionFrame();
            GetRotationFromJoints(fbx_obj.transform.GetChild(2),curr_frame);
            curr_animation.AddMotionFrame(curr_frame);
        }
        return curr_animation;
    }

    public static void GetRotationFromJoints(Transform root,MotionFrame fm){
        fm.AddJointRotation(root.name,root.localRotation);
        for(int i = 0; i < root.childCount;i++){
            GetRotationFromJoints(root.GetChild(i),fm);
        }
    }

    
}