using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/* Written by Kexin ZHENG 
* Visualize drawn line
*/
public class LineDrawer : MonoBehaviour
{
    // retrieve data
    public int END { get { return points.Count; } }
    public Vector3[] POINTS { get { return points.ToArray(); } }

    // basic settings
    [SerializeField] Material lineMat;

    private Vector3 startPos;
    private Vector3 endPos;

    private List<Vector3> points;

    private int positionCount = 0;

    private Vector3 prevPointDistance = Vector3.zero;

    private LineRenderer LineRenderer { get; set; }

    public void InitializeLine(Transform parent, Vector3 position)
    {
        positionCount = 2;

        //transform.parent = anchor?.transform ?? parent;
        transform.position = position;

        var mlinerenderer = transform.GetComponent<LineRenderer>();
        mlinerenderer.startWidth = 0.003f;
        mlinerenderer.endWidth = 0.003f;

        mlinerenderer.startColor = new Color(0.22f,0.81f,0.68f,0.8f);//Color.green;
        mlinerenderer.endColor = new Color(0.22f,0.81f,0.68f,0.8f);//Color.green;

        mlinerenderer.useWorldSpace = true;
        mlinerenderer.positionCount = positionCount;

        mlinerenderer.numCornerVertices = 5;
        mlinerenderer.numCapVertices = 5;

        mlinerenderer.SetPosition(0, position);
        mlinerenderer.SetPosition(1, position);

        LineRenderer = mlinerenderer;

        startPos = position;
        points = new List<Vector3>();
        prevPointDistance = startPos;
        points.Add(startPos);
    }

    public void AddPoint(Vector3 position)
    {
        if (prevPointDistance == null)
            prevPointDistance = position;

        if (prevPointDistance != null && Mathf.Abs(Vector3.Distance(prevPointDistance, position)) >= 0.006f)
        {
            prevPointDistance = position;
            positionCount++;

            LineRenderer.positionCount = positionCount;

            // index 0 positionCount must be - 1
            LineRenderer.SetPosition(positionCount - 1, position);
            points.Add(position);
            endPos = position;
        }
    }
}
