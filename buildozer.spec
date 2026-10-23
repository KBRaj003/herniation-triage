[app]
title = Herniation Triage
package.name = herniationtriage
package.domain = org.research
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1
requirements = python3,kivy==2.3.0,numpy==1.26.4
orientation = portrait
fullscreen = 0
android.api = 33
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True
android.numeric_version = 1
android.no_compile_py = 1
android.permissions = 

[buildozer]
log_level = 2
warn_on_root = 1
