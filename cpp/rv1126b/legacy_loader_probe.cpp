// Link against the existing RV1126B ir_left_projector.cpp for a read-only smoke test.
#include "ir_left_projector.h"
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 2) return 2;
        fusion_sensor::IrLeftProjector projector;
        if (!projector.loadCalibration(argv[1])) return 1;
        std::cout << "Legacy YAML load passed; legacy projection is NOT model-compatible.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
