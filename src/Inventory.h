#pragma once
#include "Block.h"
#include <array>

class Inventory {
public:
    static constexpr int HOTBAR_SIZE = 9;

    Inventory();

    int selectedSlot() const { return m_selectedSlot; }
    void selectSlot(int slot);

    void addBlock(BlockType type);
    bool hasBlock(BlockType type) const;
    void removeBlock(BlockType type);
    int countBlock(BlockType type) const;

    BlockType getSelectedBlock() const { return m_hotbar[m_selectedSlot]; }

    const std::array<BlockType, HOTBAR_SIZE>& hotbar() const { return m_hotbar; }

    void scroll(int direction);

private:
    std::array<BlockType, HOTBAR_SIZE> m_hotbar;
    int m_selectedSlot = 0;
};
