#pragma once

#include "Vector3D.h"

#define MAX_COORD 1e7

class BoundingBox
{
    Vector3D minPoint;
    Vector3D maxPoint;

public:
    BoundingBox(const Vector3D& v1, const Vector3D& v2)
    {
        if (v1.x < v2.x)
        {
            minPoint.x = v1.x;
            maxPoint.x = v2.x;
        }
        else
        {
            minPoint.x = v2.x;
            maxPoint.x = v1.x;
        }

        if (v1.y < v2.y)
        {
            minPoint.y = v1.y;
            maxPoint.y = v2.y;
        }
        else
        {
            minPoint.y = v2.y;
            maxPoint.y = v1.y;
        }

        if (v1.z < v2.z)
        {
            minPoint.z = v1.z;
            maxPoint.z = v2.z;
        }
        else
        {
            minPoint.z = v2.z;
            maxPoint.z = v1.z;
        }
    }

    BoundingBox()
    {
        reset();
    }

    void addVertex(const Vector3D& v)
    {
        if (v.x < minPoint.x)
            minPoint.x = v.x;

        if (v.x > maxPoint.x)
            maxPoint.x = v.x;

        if (v.y < minPoint.y)
            minPoint.y = v.y;

        if (v.y > maxPoint.y)
            maxPoint.y = v.y;

        if (v.z < minPoint.z)
            minPoint.z = v.z;

        if (v.z > maxPoint.z)
            maxPoint.z = v.z;
    }

    void addVertices(const Vector3D* v, int numVertices)
    {
        for (register int i = 0; i < numVertices; i++)
        {
            if (v[i].x < minPoint.x)
                minPoint.x = v[i].x;

            if (v[i].x > maxPoint.x)
                maxPoint.x = v[i].x;

            if (v[i].y < minPoint.y)
                minPoint.y = v[i].y;

            if (v[i].y > maxPoint.y)
                maxPoint.y = v[i].y;

            if (v[i].z < minPoint.z)
                minPoint.z = v[i].z;

            if (v[i].z > maxPoint.z)
                maxPoint.z = v[i].z;
        }
    }

    bool isEmpty() const
    {
        return minPoint.x > maxPoint.x || minPoint.y > maxPoint.y || minPoint.z > maxPoint.z;
    }

    bool contains(const Vector3D& pos) const
    {
        return pos.x >= minPoint.x && pos.x <= maxPoint.x &&
            pos.y >= minPoint.y && pos.y <= maxPoint.y &&
            pos.z >= minPoint.z && pos.z <= maxPoint.z;
    }

    bool intersects(const BoundingBox& box) const
    {
        if ((maxPoint.x < box.minPoint.x) || (minPoint.x > box.maxPoint.x))
            return false;

        if ((maxPoint.y < box.minPoint.y) || (minPoint.y > box.maxPoint.y))
            return false;

        if ((maxPoint.z < box.minPoint.z) || (minPoint.z > box.maxPoint.z))
            return false;

        return true;
    }

    void reset()
    {
        minPoint.x = MAX_COORD;
        minPoint.y = MAX_COORD;
        minPoint.z = MAX_COORD;

        maxPoint.x = -MAX_COORD;
        maxPoint.y = -MAX_COORD;
        maxPoint.z = -MAX_COORD;
    }

    Vector3D getMinPoint() const
    {
        return minPoint;
    }

    Vector3D getMaxPoint() const
    {
        return maxPoint;
    }

    Vector3D getVertex(int index) const
    {
        return Vector3D(index & 1 ? maxPoint.x : minPoint.x,
            index & 2 ? maxPoint.y : minPoint.y,
            index & 4 ? maxPoint.z : minPoint.z);
    }

    Vector3D getCenter() const
    {
        return (minPoint + maxPoint) * 0.5f;
    }

    Vector3D getSize() const
    {
        return Vector3D(maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z);
    }

    void merge(const BoundingBox& box)
    {
        if (box.minPoint.x < minPoint.x)
            minPoint.x = box.minPoint.x;

        if (box.minPoint.y < minPoint.y)
            minPoint.y = box.minPoint.y;

        if (box.minPoint.z < minPoint.z)
            minPoint.z = box.minPoint.z;

        if (box.maxPoint.x > maxPoint.x)
            maxPoint.x = box.maxPoint.x;

        if (box.maxPoint.y > maxPoint.y)
            maxPoint.y = box.maxPoint.y;

        if (box.maxPoint.z > maxPoint.z)
            maxPoint.z = box.maxPoint.z;
    }

    void grow(const Vector3D& delta)
    {
        minPoint -= delta;
        maxPoint += delta;
    }

    void grow(float delta)
    {
        minPoint.x -= delta;
        minPoint.y -= delta;
        minPoint.z -= delta;
        maxPoint.x -= delta;
        maxPoint.y -= delta;
        maxPoint.z -= delta;
    }
};
