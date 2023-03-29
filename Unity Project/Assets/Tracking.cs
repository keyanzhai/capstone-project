using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Tracking : MonoBehaviour
{   
    public UDPReceive udpReceive;
    public GameObject[] handPoints;
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        string data = udpReceive.data;
        data = data.Remove(0, 1);
        data = data.Remove(data.Length-1, 1);
        print(data);
        string[] points = data.Split(',');
        print(points[0]);

        float x = float.Parse(points[0]);
        float y = float.Parse(points[1]);
        float z = float.Parse(points[2]);
        handPoints[0].transform.position = new Vector3(x, y, z);
    }
}
// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;

// public class Tracking : MonoBehaviour
// {
//     // Start is called before the first frame update
//     public UDPReceive udpReceive;
//     public GameObject[] handPoints;
//     void Start()
//     {
        
//     }

//     // Update is called once per frame
//     void Update()
//     {
//         string data = udpReceive.data;

//         data = data.Remove(0, 1);
//         data = data.Remove(data.Length-1, 1);
//         print(data);
//         string[] points = data.Split(',');
//         print(points[0]);
//         float x = 7-float.Parse(points[0]);
//         float y = float.Parse(points[1]);
//         float z = float.Parse(points[2]);
//         handPoints[i].transform.localPosition = new Vector3(x, y, z);
//         }


//     }
// }