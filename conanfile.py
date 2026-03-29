from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy
from os.path import join


class SequenceProcessingConan(ConanFile):
    name = "sequence_processing_c"
    version = "0.0.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": True, "fPIC": True}
    exports_sources = "CMakeLists.txt", "src/*", "test/*", "resources/*"

    def layout(self):
        cmake_layout(self, src_folder=".")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "*.h", src=join(self.source_folder, "src"), dst=join(self.package_folder, "include"))
        copy(self, "*.a", src=self.build_folder, dst=join(self.package_folder, "lib"))
        copy(self, "*.so", src=self.build_folder, dst=join(self.package_folder, "lib"))
        copy(self, "*.dylib", src=self.build_folder, dst=join(self.package_folder, "lib"))
        copy(self, "*.dll", src=self.build_folder, dst=join(self.package_folder, "bin"))

    def package_info(self):
        self.cpp_info.libs = ["SequenceProcessing"]

