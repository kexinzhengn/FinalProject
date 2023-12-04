using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
/* Written by Kexin ZHENG 
* Array operations
* Line geometry processors
*/
public static class SystemUtils{

    // Find value in array //
    public static bool IsInArray<T>(this T[] arr, T val){
        int pos = Array.IndexOf(arr, val);
        if (pos > -1)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    // Get sub array //
    public static T[] SubArray<T>(this T[] array, int offset, int length)
	{
		T[] result = new T[length];
		Array.Copy(array, offset, result, 0, length);
		return result;
	}

    // Get sub array with start and end index //
    public static T[] SubArraySE<T>(this T[] array, int sid, int eid)
	{
        var length = eid-sid+1;
		T[] result = new T[length];
		Array.Copy(array, sid, result, 0, length);
		return result;
	}

    // Compute the arc length given a set of points //
    public static float ComputeArcLength(Vector3[] pts){
        float arc_len = 0.0f;
        for(int i = 0; i < pts.Length-1;i++){
            arc_len += Vector3.Distance(pts[i],pts[i+1]);
        }
        return arc_len;
    }

    // Convert Vector3 array in to Vector2 array //
    public static List<Vector2> Vector3To2(Vector3[] p3ds){
        var res = new List<Vector2>();
        for(int i = 0; i < p3ds.Length;i++){
            var curr_p3d = p3ds[i];
            var curr_p2d = new Vector2(curr_p3d.x,curr_p3d.y);
            res.Add(curr_p2d);
        }
        return res;
    }

    // resample points to eual distance //
    public static Vector3[] ResamplePoints(Vector3[] pts, float intvl_thres){
        // Interpolate old points to new points
        List<Vector3> new_points = new List<Vector3>();
        new_points.Add(pts[0]);
        for (int i = 1; i < pts.Length; i++)
        {
            Vector3 currentPoint = pts[i];
            Vector3 previousPoint = new_points[new_points.Count-1];
            float segmentLength = Vector3.Distance(previousPoint, currentPoint);
            float length = intvl_thres;
            float overshot = segmentLength - intvl_thres;
            while (overshot > intvl_thres)
            {
                float frac = length/segmentLength;
                Vector3 newPoint = Vector3.Lerp(previousPoint, currentPoint, frac);
                new_points.Add(newPoint);
                overshot = segmentLength - length;
                length += intvl_thres;
            }
        }
        for (int i = 0; i < new_points.Count-1;i++){
            float dist = Vector3.Distance(new_points[i+1], new_points[i]);
        }
        return new_points.ToArray();
    }

    // Get file paths inside a folder with target extension //
    public static string[] GetFilesInFolder(string folder_path, string extension){
        List<string> files = new List<string>();

        string[] paths = Directory.GetFiles(folder_path);
        foreach(var item in paths){
            string file_extension = Path.GetExtension(item).ToLower();
            if(file_extension == extension){
                files.Add(item);
            }
        }
        return files.ToArray();
    }

    //// Mid point smoothing /////
    public static Vector3[] MidpointSmoothing(Vector3[] pts)
    {
        var result = new List<Vector3>();
        result.Add(pts[0]);
        for (int i = 0; i < pts.Length - 1; i++)
        {
            var px = (pts[i].x + pts[(i + 1) % pts.Length].x) / 2;
            var py = (pts[i].y + pts[(i + 1) % pts.Length].y) / 2;
            var pz = (pts[i].z + pts[(i + 1) % pts.Length].z) / 2;
            result.Add(new Vector3(px, py,pz));
        }
        result.Add(pts[pts.Length-1]);
        return result.ToArray();
    }

    public static int FindPointIndexWithHeight(Vector3[] pts, float target_height){
        int res = 0;
        float min_cost = float.MaxValue;
        for(int i = 0; i < pts.Length;i++){
            var curr_cost = Mathf.Abs(pts[i].y - target_height);
            if(curr_cost < min_cost){
                min_cost = curr_cost;
                res = i;
            }
        }
        return res;
    }


    public static Vector3 FindPointWithHeight(Vector3[] pts, float target_height){
        int res = 0;
        float min_cost = float.MaxValue;
        for(int i = 0; i < pts.Length;i++){
            var curr_cost = Mathf.Abs(pts[i].y - target_height);
            if(curr_cost < min_cost){
                min_cost = curr_cost;
                res = i;
            }
        }
        return pts[res];
    }


    /* 
    * Code copied from function douglasPeuckerReduction, found at:
    * https://www.codeproject.com/Articles/18936/A-C-Implementation-of-Douglas-Peucker-Line-Appro
    * Modified by ZHENG Kexin
    */
    public static List<Vector3> DouglasPeuckerReduction(List<Vector2> Points, float Tolerance)
    {
        int firstPoint = 0;
        int lastPoint = Points.Count - 1;
        List<int> pointIndexsToKeep = new List<int>();

        //Add the first and last index to the keepers
        pointIndexsToKeep.Add(firstPoint);
        pointIndexsToKeep.Add(lastPoint);

        //The first and the last point cannot be the same
        while (Points[firstPoint].Equals(Points[lastPoint]))
        {
            lastPoint--;
        }

        DouglasPeuckerReduction(Points, firstPoint, lastPoint,
        Tolerance, ref pointIndexsToKeep);

        List<Vector3> returnPoints = new List<Vector3>();
        pointIndexsToKeep.Sort();
        foreach (int index in pointIndexsToKeep)
        {
            var new_pt = new Vector3(Points[index].x,Points[index].y,0.0f);
            returnPoints.Add(new_pt);
        }

        return returnPoints;
    }

    private static void DouglasPeuckerReduction(List<Vector2>points, int firstPoint, int lastPoint, float tolerance,ref List<int> pointIndexsToKeep)
    {
        float maxDistance = 0;
        int indexFarthest = 0;

        for (int index = firstPoint; index < lastPoint; index++)
        {
            float distance = PerpendicularDistance
                (points[firstPoint], points[lastPoint], points[index]);
            if (distance > maxDistance)
            {
                maxDistance = distance;
                indexFarthest = index;
            }
        }

        if (maxDistance > tolerance && indexFarthest != 0)
        {
            //Add the largest point that exceeds the tolerance
            pointIndexsToKeep.Add(indexFarthest);
            DouglasPeuckerReduction(points, firstPoint,
            indexFarthest, tolerance, ref pointIndexsToKeep);
            DouglasPeuckerReduction(points, indexFarthest,
            lastPoint, tolerance, ref pointIndexsToKeep);
        }
    }

    public static float PerpendicularDistance(Vector2 Point1, Vector2 Point2, Vector2 Point)
    {
        //Area = |(1/2)(x1y2 + x2y3 + x3y1 - x2y1 - x3y2 - x1y3)|   *Area of triangle
        //Base = v((x1-x2)²+(x1-x2)²)                               *Base of Triangle*
        //Area = .5*Base*H                                          *Solve for height
        //Height = Area/.5/Base

        float area = Mathf.Abs(.5f * (Point1.x * Point2.y + Point2.x *
        Point.y + Point.x * Point1.y - Point2.x * Point1.y - Point.x *
        Point2.y - Point1.x * Point.y));
        float bottom = Mathf.Sqrt(Mathf.Pow(Point1.x - Point2.x, 2) +
        Mathf.Pow(Point1.y - Point2.y, 2));
        float height = area / bottom * 2;

        return height;

    }

}