#include "Inventory.h"

Inventory::Inventory() {
    m_hotbar.fill(BlockType::Air);
    m_hotbar[0] = BlockType::Dirt;
    m_hotbar[1] = BlockType::Stone;
    m_hotbar[2] = BlockType::Cobblestone;
    m_hotbar[3] = BlockType::Planks;
    m_hotbar[4] = BlockType::OakLog;
    m_hotbar[5] = BlockType::Sand;
    m_hotbar[6] = BlockType::Gravel;
}

void Inventory::selectSlot(int slot) {
    if (slot >= 0 && slot < HOTBAR_SIZE) {
        m_selectedSlot = slot;
    }
}

void Inventory::scroll(int direction) {
    int newSlot = m_selectedSlot + direction;
    if (newSlot < 0) newSlot = HOTBAR_SIZE - 1;
    if (newSlot >= HOTBAR_SIZE) newSlot = 0;
    m_selectedSlot = newSlot;
}

void Inventory::addBlock(BlockType type) {
    if (type == BlockType::Air) return;
    for (int i = 0; i < HOTBAR_SIZE; ++i) {
        if (m_hotbar[i] == type || m_hotbar[i] == BlockType::Air) {
            m_hotbar[i] = type;
            return;
        }
    }
}

bool Inventory::hasBlock(BlockType type) const {
    return countBlock(type) > 0;
}

void Inventory::removeBlock(BlockType type) {
    for (int i = 0; i < HOTBAR_SIZE; ++i) {
        if (m_hotbar[i] == type) {
            m_hotbar[i] = BlockType::Air;
            return;
        }
    }
}

int Inventory::countBlock(BlockType type) const {
    int count = 0;
    for (int i = 0; i < HOTBAR_SIZE; ++i) {
        if (m_hotbar[i] == type) ++count;
    }
    return count;
}
