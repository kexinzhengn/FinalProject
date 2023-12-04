using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Xml;
using System.IO;
using System;
using System.Linq;

/*
* Written by ZHENG Kexin
* Self-defined class to store information of motion frames for modification
*/
public class MotionFrame{
    public string[] JNAME{get{return joint_names.ToArray();}}
    public int JOINT_NUM{get{return joint_names.Count;}}
    public Quaternion[] ROTS{get{return joint_rots.ToArray();}}

    Dictionary<string, Quaternion> rot_dict;
    List<string> joint_names;
    List<Quaternion> joint_rots;

    public MotionFrame(){
        joint_names = new List<string>();
        joint_rots = new List<Quaternion>();
        rot_dict = new Dictionary<string, Quaternion>();
    }

    public void AddJointRotation(string joint_name,Quaternion rot){
        joint_names.Add(joint_name);
        joint_rots.Add(rot);
        rot_dict.Add(joint_name,rot);
    }

    public Quaternion GetJointRotation(string jname){
        return rot_dict[jname];
    }

    public MotionFrame ShallowCopy(){
        return (MotionFrame)this.MemberwiseClone();
    }

}


public class MotionAnimation{
    public int FRAME_NUM{get{return frames.Count;}}
    public MotionFrame[] FRAMES{get{return frames.ToArray();}}
    
    string motion_name;
    List<MotionFrame> frames;

    public MotionAnimation(string name){
        motion_name = name;
        frames = new List<MotionFrame>();
    }

    public void AddMotionFrame(MotionFrame fm){
        frames.Add(fm);
    }

    public void AddMotionFrames(MotionFrame[] fms){
        frames.AddRange(fms);
    }

    public MotionFrame GetFrame(int idx){
        return frames[idx];
    }

    public MotionAnimation RepeatMotions(int target_length){
        var repeat_time = (int)Mathf.Floor(target_length/FRAME_NUM);
        var new_motion = new MotionAnimation(motion_name);
        for(int i = 0; i < repeat_time;i++){
            new_motion.AddMotionFrames(frames.ToArray());
        }
        var num_frame_left = target_length - FRAME_NUM * repeat_time;
        for(int i = 0; i < num_frame_left;i++){
            new_motion.AddMotionFrame(frames[i]);
        }
        return new_motion;
    }

    // sid midx eid depends on jump type
    // al and dl depsends on start and end position;
    public MotionAnimation ScaleJumpAnimation(int ascendLength, int decendLength, int sid,int midx, int eid){
        var new_frame_num = ascendLength + decendLength;
        var new_am = new MotionAnimation(motion_name+(new_frame_num).ToString());
        //anticipation
        MotionFrame[] anti_frames = SystemUtils.SubArray(FRAMES,0,sid);
        new_am.AddMotionFrames(anti_frames);

        //1. aescend
        float as_step = (float)((midx-sid+1)/(float)(ascendLength-1.0f));
        for(int i = 0; i< ascendLength;i++){
            float curr_frame_f = (float)sid + as_step * (float)i;
            var curr_frame = GetInterpolationFrame(curr_frame_f);
            new_am.AddMotionFrame(curr_frame);
        }

        //2. descend
        float ds_step = (float)((eid-midx+1.0f)/(float)(decendLength-1.0f));
        for(int i = 0; i<decendLength;i++){
            float curr_frame_f = (float)midx + ds_step * (float)i;
            var curr_frame = GetInterpolationFrame(curr_frame_f);
            new_am.AddMotionFrame(curr_frame);
        }

        // landing
        MotionFrame[] land_frames = SystemUtils.SubArraySE(FRAMES,eid+1,FRAME_NUM-1);
        new_am.AddMotionFrames(land_frames);

        return new_am;
    }





    // align old animation with new timescale
    // mainly for jumping animation
    public MotionAnimation ScaledAnimation(int new_frame_num){
        var joint_names = frames[0].JNAME;
        var new_am = new MotionAnimation(motion_name+(new_frame_num).ToString());
        for(int j = 0; j < new_frame_num; j++){
            float curr_frame_f = (float)j/(float)(new_frame_num-1) * (float)FRAME_NUM;
            float prev_frame_f = Mathf.Floor(curr_frame_f);
            float next_frame_f = Mathf.Floor(curr_frame_f+1.0f);
            if(next_frame_f<FRAME_NUM){
                var prev_frame = frames[(int)prev_frame_f];
                var next_frame = frames[(int)next_frame_f];
                var lerp_val = curr_frame_f - prev_frame_f;
                var new_fm = GetInterpolationFrame(prev_frame,next_frame,lerp_val);
                new_am.AddMotionFrame(new_fm);
            }else{
                var new_fm = this.GetFrame(this.FRAME_NUM-1).ShallowCopy();
                new_am.AddMotionFrame(new_fm);
            }
        }
        return new_am;
    }

    public MotionFrame GetInterpolationFrame(MotionFrame fm1,MotionFrame fm2,float lerp_val){
        MotionFrame curr_frame = new MotionFrame();
        var joint_names = fm1.JNAME;
        for(int i = 0; i < joint_names.Length;i++){
            Quaternion prev_quat = fm1.GetJointRotation(joint_names[i]);
            Quaternion next_quat = fm2.GetJointRotation(joint_names[i]);
            Quaternion curr_quat = Quaternion.Lerp(prev_quat,next_quat,lerp_val);
            curr_frame.AddJointRotation(joint_names[i],curr_quat);
        }
        return curr_frame;
    }

    // e.g. between frame 1 and 2. curr_frame_f could be 1.5
    public MotionFrame GetInterpolationFrame(float curr_frame_f){ 
        MotionFrame curr_frame = new MotionFrame();
        float prev_frame_f = Mathf.Floor(curr_frame_f);
        float next_frame_f = Mathf.Floor(curr_frame_f+1.0f);
        var lerp_val = curr_frame_f - prev_frame_f;
        if(next_frame_f>=this.FRAME_NUM) {
            return frames.Last().ShallowCopy();
        }else{
            var fm1 = frames[(int)prev_frame_f];
            var fm2 = frames[(int)next_frame_f];
            var joint_names = fm1.JNAME;
            for(int i = 0; i < joint_names.Length;i++){
                Quaternion prev_quat = fm1.GetJointRotation(joint_names[i]);
                Quaternion next_quat = fm2.GetJointRotation(joint_names[i]);
                Quaternion curr_quat = Quaternion.Lerp(prev_quat,next_quat,lerp_val);
                curr_frame.AddJointRotation(joint_names[i],curr_quat);
            }
            return curr_frame;
        }
    }



}
