#pragma once
#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

class Noise {
public:
    Noise(uint32_t seed = 1337) {
        std::array<int, 256> base;
        std::iota(base.begin(), base.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(base.begin(), base.end(), rng);
        for (int i = 0; i < 256; ++i) p[i] = p[i + 256] = base[i];
    }

    float perlin(float x, float y) const {
        int X = (int)std::floor(x) & 255;
        int Y = (int)std::floor(y) & 255;
        x -= std::floor(x);
        y -= std::floor(y);
        float u = fade(x), v = fade(y);
        int A = p[X] + Y, B = p[X + 1] + Y;
        return lerp(v,
            lerp(u, grad(p[A],     x,     y), grad(p[B],     x - 1, y)),
            lerp(u, grad(p[A + 1], x, y - 1), grad(p[B + 1], x - 1, y - 1)));
    }

    float perlin3D(float x, float y, float z) const {
        int X = (int)std::floor(x) & 255;
        int Y = (int)std::floor(y) & 255;
        int Z = (int)std::floor(z) & 255;
        x -= std::floor(x);
        y -= std::floor(y);
        z -= std::floor(z);
        float u = fade(x), v = fade(y), w = fade(z);
        int A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z;
        int B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;
        return lerp(w, lerp(v,
            lerp(u, grad3(p[AA],     x,     y,     z),     grad3(p[BA],     x - 1, y,     z)),
            lerp(u, grad3(p[AB],     x,     y - 1, z),     grad3(p[BB],     x - 1, y - 1, z))),
            lerp(v,
            lerp(u, grad3(p[AA + 1], x,     y,     z - 1), grad3(p[BA + 1], x - 1, y,     z - 1)),
            lerp(u, grad3(p[AB + 1], x,     y - 1, z - 1), grad3(p[BB + 1], x - 1, y - 1, z - 1))));
    }

    float fbm(float x, float y, int octaves = 6,
              float persistence = 0.5f, float lacunarity = 2.0f) const {
        float val = 0, amp = 1, freq = 1, maxVal = 0;
        for (int i = 0; i < octaves; ++i) {
            val    += perlin(x * freq, y * freq) * amp;
            maxVal += amp;
            amp    *= persistence;
            freq   *= lacunarity;
        }
        return val / maxVal;
    }

    float fbm3D(float x, float y, float z, int octaves = 3,
                float persistence = 0.5f, float lacunarity = 2.0f) const {
        float val = 0, amp = 1, freq = 1, maxVal = 0;
        for (int i = 0; i < octaves; ++i) {
            val    += perlin3D(x * freq, y * freq, z * freq) * amp;
            maxVal += amp;
            amp    *= persistence;
            freq   *= lacunarity;
        }
        return val / maxVal;
    }

    static float hash(float x, float y) {
        float n = std::sin(x * 12.9898f + y * 78.233f) * 43758.5453f;
        return n - std::floor(n);
    }

    static float hash(float x, float y, float z) {
        float n = std::sin(x * 12.9898f + y * 78.233f + z * 37.719f) * 43758.5453f;
        return n - std::floor(n);
    }

private:
    std::array<int, 512> p;

    static float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    static float lerp(float t, float a, float b) { return a + t * (b - a); }
    static float grad(int h, float x, float y) {
        switch (h & 7) {
            case 0: return  x + y; case 1: return -x + y;
            case 2: return  x - y; case 3: return -x - y;
            case 4: return  x;     case 5: return -x;
            case 6: return  y;     case 7: return -y;
            default: return 0;
        }
    }
    static float grad3(int h, float x, float y, float z) {
        switch (h & 15) {
            case  0: return  x + y; case  1: return -x + y;
            case  2: return  x - y; case  3: return -x - y;
            case  4: return  x + z; case  5: return -x + z;
            case  6: return  x - z; case  7: return -x - z;
            case  8: return  y + z; case  9: return -y + z;
            case 10: return  y - z; case 11: return -y - z;
            case 12: return  y + x; case 13: return -y + z;
            case 14: return -y - x; case 15: return  y - z;
            default: return 0;
        }
    }
};
