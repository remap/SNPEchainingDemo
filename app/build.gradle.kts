import java.util.Properties
val localProperties = Properties().apply {
    val file = rootProject.file("local.properties")
    if (file.exists()) load(file.inputStream())
}

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

        // Inject secrets into a generated BuildConfig class
        buildConfigField("String", "HF_TOKEN", "\"${localProperties.getProperty("HF_READ_TOKEN")}\"")
        buildConfigField("String", "HF_REPO", "\"uclaremap/pose_estimation\"")
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
        buildConfig = true
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
        noCompress += listOf("tflite")
        noCompress += listOf("onnx")
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

    implementation("com.google.android.gms:play-services-base:18.3.0")
    implementation("com.google.android.gms:play-services-tflite-java:16.1.0") // Often needed for the underlying runtime
    // Google ML Kit for Subject Segmentation (Background Removal)
    implementation("com.google.android.gms:play-services-mlkit-subject-segmentation:16.0.0-beta1")
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.1")
    // Core Android KTX for bitmap handling
    implementation("androidx.core:core-ktx:1.12.0")

    implementation("io.getstream:photoview:1.0.3")
    implementation("com.airbnb.android:lottie:6.3.0")

    // Networking and JSON
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.code.gson:gson:2.10.1")

    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
}