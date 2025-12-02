plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.snpechainingdemo"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.snpechainingdemo"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    packagingOptions {
        // Include shared libc++ shipped with Android NDK
        pickFirst("lib/**/libc++_shared.so")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    ndkVersion = "26.2.11394342"
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        viewBinding = true
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
    // make Gradle pick up src/main/jniLibs and src/main/assets
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
            assets.srcDirs("src/main/assets")
        }
    }
    androidResources {
        // keep DLC entries stored (STORED), not DEFLATED, so we can mmap them
        noCompress += listOf("dlc")
    }

}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // Core tokenizer API (Rust tokenizer via JNI)
    implementation("ai.djl.huggingface:tokenizers:0.34.0")

    // Android native .so for the tokenizer (packaged as an AAR)
    implementation("ai.djl.android:tokenizer-native:0.33.0"){
        // Make sure libc++_shared gets pulled in
        // If missing, add a direct dependency:
        implementation("org.bytedeco:javacpp:1.5.9")
    }

    implementation("io.getstream:photoview:1.0.3")
    implementation("com.airbnb.android:lottie:6.3.0")
}