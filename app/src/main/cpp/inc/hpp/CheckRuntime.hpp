//==============================================================================
//
//  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef CHECKRUNTIME_H
#define CHECKRUNTIME_H

#pragma once

#include "SNPE/SNPEFactory.hpp"

//zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime);
bool checkGLCLInteropSupport();

inline zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime) {
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
        runtime = zdl::DlSystem::Runtime_t::GPU;
        if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
            runtime = zdl::DlSystem::Runtime_t::CPU;
        }
    }
    return runtime;
}

#endif