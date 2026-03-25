#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// Every block type in Ferox
enum class BlockType : uint8_t {
    Air         = 0,
    Grass       = 1,
    Dirt        = 2,
    Stone       = 3,
    Cobblestone = 4,
    Planks      = 5,
    Sand        = 6,
    Water       = 7,
    Gravel      = 8,
    OakLog      = 9,
    OakLeaves   = 10,
    CoalOre     = 11,
    IronOre     = 12,
    GoldOre     = 13,
    DiamondOre  = 14,
    Bedrock     = 15,
    BirchLog    = 16,
    BirchLeaves = 17,
    TallGrass   = 18,
};

inline bool isBlockSolid(BlockType b) {
    return b != BlockType::Air && b != BlockType::Water && b != BlockType::OakLeaves 
           && b != BlockType::BirchLeaves && b != BlockType::TallGrass;
}

inline bool isBlockOpaque(BlockType b) {
    return b != BlockType::Air && b != BlockType::Water && b != BlockType::OakLeaves
           && b != BlockType::BirchLeaves && b != BlockType::TallGrass;
}

inline bool isBlockTransparent(BlockType b) {
    return b == BlockType::Air || b == BlockType::Water || b == BlockType::OakLeaves
           || b == BlockType::BirchLeaves || b == BlockType::TallGrass;
}

// Texture layer indices in the texture array
// faceDir: 0=top, 1=bottom, 2=north(+Z), 3=south(-Z), 4=east(+X), 5=west(-X)
inline int blockTexture(BlockType type, int faceDir) {
    switch (type) {
        case BlockType::Grass:
            if (faceDir == 0) return 0;
            if (faceDir == 1) return 2;
            return 1;
        case BlockType::Dirt:        return 2;
        case BlockType::Stone:       return 3;
        case BlockType::Cobblestone: return 4;
        case BlockType::Planks:      return 5;
        case BlockType::Sand:        return 6;
        case BlockType::Water:       return 7;
        case BlockType::Gravel:      return 8;
        case BlockType::OakLog:
        case BlockType::BirchLog:
        case BlockType::OakLeaves:
        case BlockType::BirchLeaves:
        case BlockType::CoalOre:
        case BlockType::IronOre:
        case BlockType::GoldOre:
        case BlockType::DiamondOre:
        case BlockType::Bedrock:
        case BlockType::TallGrass:   return 3;
        default:                     return 3;
    }
}

inline float blockShade(int faceDir) {
    switch (faceDir) {
        case 0:  return 1.00f; // top
        case 1:  return 0.50f; // bottom
        case 2:  return 0.80f; // north
        case 3:  return 0.80f; // south
        case 4:  return 0.65f; // east
        default: return 0.65f; // west
    }
}
