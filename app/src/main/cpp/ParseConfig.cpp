//
// Created by Chiheb Boussema on 22/9/25.
//
// ----- ParseConfig.cpp  -----
#include <string>
#include <unordered_map>
#include <vector>
#include <cctype>
#include <stdexcept>
#include <android/log.h>
#include "ParseConfig.hpp"

#define LOG_TAG "PARSE_CONFIG"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

// Minimal JSON tokenizer/parser for a restricted subset
namespace minijson {

    struct Cursor {
        const std::string* s = nullptr;
        size_t i = 0;
        void init(const std::string& str){ s=&str; i=0; }
        bool end() const { return i >= s->size(); }
        char peek() const { return end()? '\0' : (*s)[i]; }
        char get() { return end()? '\0' : (*s)[i++]; }
        void skipWS(){ while(!end() && std::isspace((unsigned char)peek())) ++i; }
    };

    static bool expect(Cursor& c, char ch, std::string* emsg) {
        c.skipWS();
        if (c.end() || c.peek()!=ch) { if (emsg) *emsg = std::string("Expected '")+ch+"'"; return false; }
        ++c.i; return true;
    }

    // --- number parsing helpers (simple/robust enough for JSON numbers) ---
    static bool parseNumber(Cursor& c, double& out, std::string* emsg) {
        c.skipWS();
        size_t start = c.i;
        if (c.end()) { if (emsg) *emsg = "Expected number"; return false; }

        // valid chars for numbers in JSON: + - . digits e E
        auto isNumChar = [](char ch)->bool {
            return (ch=='+' || ch=='-' || ch=='.' || ch=='e' || ch=='E' || (ch>='0' && ch<='9'));
        };
        if (!isNumChar(c.peek())) { if (emsg) *emsg = "Expected number"; return false; }

        while (!c.end() && isNumChar(c.peek())) ++c.i;

        // strtod on substring
        const char* begin = c.s->c_str() + start;
        char* endptr = nullptr;
        out = std::strtod(begin, &endptr);
        if (endptr != (c.s->c_str() + c.i)) {
            if (emsg) *emsg = "Invalid number";
            return false;
        }
        return true;
    }

    static bool parseUInt(Cursor& c, uint32_t& out, std::string* emsg) {
        double d = 0.0;
        if (!parseNumber(c, d, emsg)) return false;
        if (d < 0.0 || d > static_cast<double>(UINT32_MAX)) {
            if (emsg) *emsg = "Unsigned integer out of range";
            return false;
        }
        out = static_cast<uint32_t>(d);
        return true;
    }

    static bool parseString(Cursor& c, std::string& out, std::string* emsg) {
        c.skipWS();
        if (c.peek() != '\"') { if (emsg) *emsg = "Expected string"; return false; }
        ++c.i; // skip opening "
        out.clear();
        while (!c.end()) {
            char ch = c.get();
            if (ch == '\"') return true;       // closing "
            if (ch == '\\') {                  // very simple escape handling (\" and \\ only)
                if (c.end()) { if (emsg) *emsg = "Bad escape"; return false; }
                char e = c.get();
                if (e=='\"' || e=='\\') out.push_back(e);
                else { if (emsg) *emsg = "Unsupported escape"; return false; }
            } else {
                out.push_back(ch);
            }
        }
        if (emsg) *emsg = "Unterminated string";
        return false;
    }

    static bool parseStringObject(Cursor& c, std::unordered_map<std::string,std::string>& out, std::string* emsg) {
        // Expects: { "k":"v", ... }
        out.clear();
        if (!expect(c,'{',emsg)) return false;
        c.skipWS();
        if (!c.end() && c.peek()=='}') { ++c.i; return true; } // empty
        while (true) {
            std::string key, val;
            if (!parseString(c,key,emsg)) return false;
            if (!expect(c,':',emsg)) return false;
            if (!parseString(c,val,emsg)) return false;
            out.emplace(std::move(key), std::move(val));
            c.skipWS();
            if (!c.end() && c.peek()==',') { ++c.i; continue; }
            if (!expect(c,'}',emsg)) return false;
            break;
        }
        return true;
    }

    static bool parseModelObject(Cursor& c, ModelCfg& m, std::string* emsg) {
        // Expects: { "name": "...", "asset": "...", ["runtime":"D"], "inputs": {...}, "outputs": {...} }
        if (!expect(c,'{',emsg)) return false;

        bool haveName=false, haveAsset=false, haveInputs=false, haveOutputs=false;
        m = ModelCfg{}; // reset

        while (true) {
            c.skipWS();
            if (c.end()) { if (emsg) *emsg="Unterminated object"; return false; }
            if (c.peek()=='}') { ++c.i; break; }

            std::string key;
            if (!parseString(c,key,emsg)) return false;
            if (!expect(c,':',emsg)) return false;

            if (key=="name") {
                if (!parseString(c,m.name,emsg)) return false;
                haveName=true;
            } else if (key=="asset") {
                if (!parseString(c,m.asset,emsg)) return false;
                haveAsset=true;
            } else if (key=="runtime") {
                std::string r;
                if (!parseString(c,r,emsg)) return false;
                m.runtime = r.empty()? 0 : r[0];
            } else if (key=="inputs") {
                if (!parseStringObject(c, m.inputs, emsg)) return false;
                haveInputs = true;
            } else if (key == "inputEncodings") {
                if (!parseStringObject(c, m.inputEncodings, emsg)) return false;
                haveInputs = true;
            } else if (key=="outputs") {
                if (!parseStringObject(c, m.outputs, emsg)) return false;
                haveOutputs=true;
            } else {
                // skip value (string or object or array) — but we only need str/object here
                // try string first
                c.skipWS();
                if (c.peek()=='\"') {
                    std::string dummy;
                    if (!parseString(c,dummy,emsg)) return false;
                } else if (c.peek()=='{') {
                    std::unordered_map<std::string,std::string> dummy;
                    if (!parseStringObject(c,dummy,emsg)) return false;
                } else if (c.peek()=='[') {
                    // skip array of strings/objects (not expected)
                    if (!expect(c,'[',emsg)) return false;
                    int depth=1;
                    while(!c.end() && depth>0){
                        char ch=c.get();
                        if (ch=='\"'){ std::string tmp; c.i--; if(!parseString(c,tmp,emsg)) return false; }
                        else if (ch=='[') depth++;
                        else if (ch==']') depth--;
                    }
                    if (depth!=0){ if(emsg)*emsg="Unterminated array"; return false; }
                } else {
                    if (emsg) *emsg = "Unsupported value for key '"+key+"'";
                    return false;
                }
            }

            c.skipWS();
            if (!c.end() && c.peek()==',') { ++c.i; continue; }
            if (!c.end() && c.peek()=='}') { ++c.i; break; }
            if (c.end()) { if (emsg) *emsg="Unterminated object"; return false; }
            // else continue loop
        }

        if (!haveName || !haveAsset || !haveInputs || !haveOutputs) {
            if (emsg) *emsg = "Model object missing required fields (name/asset/inputs/outputs)";
            return false;
        }
        return true;
    }

    // --- InitSpec parsing ---
    static InitKind kindFromString(const std::string& s) {
        if (s == "zero")       return InitKind::ZERO;
        if (s == "random")     return InitKind::RANDOM;
        if (s == "file")       return InitKind::FILE_PATH;
        if (s == "asset")      return InitKind::ASSET_PATH;
        if (s == "const")      return InitKind::CONST_VALUE;
        return InitKind::UNKNOWN;
    }

    // Parse: { "kind":"random" | "zero" | "file" | "asset" | "const",
    //          "path":"...", "mean":..., "std":..., "seed":..., "value":... }
    static bool parseInitSpecObject(Cursor& c, InitSpec& spec, std::string* emsg) {
        if (!expect(c,'{',emsg)) return false;
        spec = InitSpec{};

        while (true) {
            c.skipWS();
            if (c.end()) { if (emsg) *emsg="Unterminated init object"; return false; }
            if (c.peek()=='}') { ++c.i; break; }

            std::string key;
            if (!parseString(c, key, emsg)) return false;
            if (!expect(c, ':', emsg)) return false;

            if (key == "kind") {
                std::string k;
                if (!parseString(c, k, emsg)) return false;
                spec.kind = kindFromString(k);
            } else if (key == "path") {
                if (!parseString(c, spec.path, emsg)) return false;
            } else if (key == "mean") {
                double d=0.0; if (!parseNumber(c, d, emsg)) return false; spec.mean = static_cast<float>(d);
            } else if (key == "std") {
                double d=1.0; if (!parseNumber(c, d, emsg)) return false; spec.std = static_cast<float>(d);
            } else if (key == "seed") {
                uint32_t u=0; if (!parseUInt(c, u, emsg)) return false; spec.seed = u;
            } else if (key == "value") {
                double d=0.0; if (!parseNumber(c, d, emsg)) return false; spec.value = static_cast<float>(d);
            } else {
                // skip unknown simple value (string/number/object/array not expected here)
                c.skipWS();
                if (c.peek()=='\"') {
                    std::string dummy; if (!parseString(c, dummy, emsg)) return false;
                } else if (c.peek()=='{') {
                    // skip nested object generically
                    if (!expect(c,'{',emsg)) return false;
                    int depth=1;
                    while(!c.end() && depth>0){
                        char ch=c.get();
                        if (ch=='\"'){ std::string tmp; c.i--; if(!parseString(c,tmp,emsg)) return false; }
                        else if (ch=='{') depth++;
                        else if (ch=='}') depth--;
                    }
                    if (depth!=0){ if(emsg)*emsg="Unterminated object"; return false; }
                } else if (c.peek()=='[') {
                    if (!expect(c,'[',emsg)) return false;
                    int depth=1;
                    while(!c.end() && depth>0){
                        char ch=c.get();
                        if (ch=='\"'){ std::string tmp; c.i--; if(!parseString(c,tmp,emsg)) return false; }
                        else if (ch=='[') depth++;
                        else if (ch==']') depth--;
                    }
                    if (depth!=0){ if(emsg)*emsg="Unterminated array"; return false; }
                } else {
                    // maybe a number
                    double dummyN; if (!parseNumber(c, dummyN, nullptr)) {
                        if (emsg) *emsg = "Unsupported value in init spec at key '"+key+"'";
                        return false;
                    }
                }
            }

            c.skipWS();
            if (!c.end() && c.peek()==',') { ++c.i; continue; }
            if (!c.end() && c.peek()=='}') { ++c.i; break; }
        }

        return true;
    }

    // Parse: "init": { "wsTensorName": { ... InitSpec... }, ... }
    static bool parseInitMap(Cursor& c, std::unordered_map<std::string, InitSpec>& out, std::string* emsg) {
        out.clear();
        if (!expect(c,'{',emsg)) return false;
        c.skipWS();
        if (!c.end() && c.peek()=='}') { ++c.i; return true; } // empty

        while (true) {
            std::string wsName;
            if (!parseString(c, wsName, emsg)) return false;
            if (!expect(c, ':', emsg)) return false;

            InitSpec spec;
            if (!parseInitSpecObject(c, spec, emsg)) return false;
            out.emplace(std::move(wsName), std::move(spec));

            c.skipWS();
            if (!c.end() && c.peek()==',') { ++c.i; continue; }
            if (!expect(c, '}', emsg)) return false;
            break;
        }
        return true;
    }


    static bool parsePipeline(Cursor& c, PipelineCfg& cfg, std::string* emsg) {
        // Expects top-level object with "models":[ ... ]
        if (!expect(c,'{',emsg)) return false;

        bool haveModels=false;
//        std::string baseDir;
        while (true) {
            c.skipWS();
            if (c.end()) { if (emsg)*emsg="Unterminated top object"; return false; }
            if (c.peek()=='}') { ++c.i; break; }

            std::string key;
            if (!parseString(c,key,emsg)) return false;
            if (!expect(c,':',emsg)) return false;

            if (key=="name") {
                if (!parseString(c, cfg.name, emsg)) return false;
            } else if (key=="baseDir") {
                if (!parseString(c,cfg.baseDir,emsg)) continue;
            } else if (key=="models") {
                // parse array of model objects
                if (!expect(c,'[',emsg)) return false;
                c.skipWS();
                if (!c.end() && c.peek()==']') { // empty array
                    ++c.i;
                } else {
                    while (true) {
                        ModelCfg m;
                        if (!parseModelObject(c,m,emsg)) return false;
//                        if (haveBaseDir) m.baseDir = baseDir;
                        cfg.models.push_back(std::move(m));
                        c.skipWS();
                        if (!c.end() && c.peek()==',') { ++c.i; continue; }
                        if (!expect(c,']',emsg)) return false;
                        break;
                    }
                }
                haveModels = true;
            } else if (key=="init") {
                if (!minijson::parseInitMap(c, cfg.init, emsg)) return false;
            } else {
                // skip unknown field (string / object / array)
                c.skipWS();
                if (c.peek()=='\"') {
                    std::string dummy;
                    if (!parseString(c,dummy,emsg)) return false;
                } else if (c.peek()=='{') {
//                    std::unordered_map<std::string,std::string> dummy;
//                    if (!parseStringObject(c,dummy,emsg)) return false;
                    // skip object
                    if (!expect(c,'{',emsg)) return false;
                    int depth=1;
                    while(!c.end() && depth>0){
                        char ch=c.get();
                        if (ch=='\"'){ std::string tmp; c.i--; if(!parseString(c,tmp,emsg)) return false; }
                        else if (ch=='{') depth++;
                        else if (ch=='}') depth--;
                    }
                    if (depth!=0){ if(emsg)*emsg="Unterminated object"; return false; }
                } else if (c.peek()=='[') {
                    if (!expect(c,'[',emsg)) return false;
                    int depth=1;
                    while(!c.end() && depth>0){
                        char ch=c.get();
                        if (ch=='\"'){ std::string tmp; c.i--; if(!parseString(c,tmp,emsg)) return false; }
                        else if (ch=='[') depth++;
                        else if (ch==']') depth--;
                    }
                    if (depth!=0){ if(emsg)*emsg="Unterminated array"; return false; }
                } else {
//                    if (emsg) *emsg = "Unsupported value at top-level key '"+key+"'";
//                    return false;
                    // maybe number: consume it
                    double dummyN; if (!parseNumber(c, dummyN, nullptr)) {
                        if (emsg) *emsg = "Unsupported value in pipeline at top-level key '"+key+"'";
                        return false;
                    }
                }
            }

            c.skipWS();
            if (!c.end() && c.peek()==',') { ++c.i; continue; }
            if (!c.end() && c.peek()=='}') { ++c.i; break; }
        }
        if (!haveModels) { if (emsg) *emsg = "Pipeline missing 'models' array"; return false; }
        return true;
    }

    // Parse an array of pipeline objects: [ { ..pipeline.. }, { ..pipeline.. }, ... ]
    static bool parsePipelineArray(Cursor& c, std::vector<PipelineCfg>& out, std::string* emsg) {
        if (!expect(c,'[',emsg)) return false;
        c.skipWS();
        if (!c.end() && c.peek()==']') { ++c.i; return true; } // empty

        while (true) {
            PipelineCfg p;
            if (!parsePipeline(c, p, emsg)) return false;
            out.push_back(std::move(p));
            c.skipWS();
            if (!c.end() && c.peek()==',') { ++c.i; continue; }
            if (!expect(c,']',emsg)) return false;
            break;
        }
        return true;
    }

} // namespace minijson

// Public API you asked for:
bool ParseConfig(const std::string& json, PipelineCfg& cfg, std::string* emsg) {
    minijson::Cursor cur;
    cur.init(json);
    cur.skipWS();
    cfg = PipelineCfg{};
    if (!minijson::parsePipeline(cur, cfg, emsg)) return false;
    cur.skipWS();
    if (!cur.end()) { if (emsg) *emsg = "Trailing characters after JSON"; return false; }
    return true;
}

// Accepts:
// 1) top-level array: [ {pipeline}, {pipeline}, ... ]
// 2) top-level object with "pipelines": [ ... ]
// 3) single pipeline object { ... }  (falls back to 1 pipeline)
bool ParseMultiConfig(const std::string& json, MultiPipelinesCfg& out, std::string* emsg) {
    using namespace minijson;
    Cursor cur;
    cur.init(json);
    cur.skipWS();
    out.pipes.clear();

    if (cur.end()) { if (emsg)*emsg="Empty JSON"; return false; }

    if (cur.peek()=='[') {
        // Case (1): top-level array of pipelines
        if (!parsePipelineArray(cur, out.pipes, emsg)) return false;
    } else if (cur.peek()=='{') {
        // Could be either a single pipeline object or an object with "pipelines":[...]
        // Peek inside:
        size_t save = cur.i;
        expect(cur,'{',nullptr);
        // scan object for a "pipelines" key at top level
        bool foundPipelines = false;
        while (!cur.end() && cur.peek()!='}') {
            std::string key;
            if (!parseString(cur, key, emsg)) { cur.i = save; break; }
            if (!expect(cur, ':', emsg)) { cur.i = save; break; }

            if (key == "baseDir") {
                LOGI("[Parse Config] Found base dir!");
                if (!parseString(cur,out.baseDir,emsg)) continue;
                cur.skipWS();
                if (!cur.end() && cur.peek() == ',') { ++cur.i; }
                continue;
            } else if (key == "pipelines") {
                LOGI("[Parse Config] Found pipelines!");
                foundPipelines = true;
                if (!parsePipelineArray(cur, out.pipes, emsg)) return false;
                // skip rest of object keys (if any)
                while (!cur.end() && cur.peek()!= '}') {
                    // skip , key:value pairs generically
                    cur.skipWS();
                    if (cur.peek()==',') { ++cur.i; }
                    // read a key or close
                    cur.skipWS();
                    if (cur.peek()=='}') break;
                    std::string k2; if (!parseString(cur,k2,emsg)) return false;
                    if (!expect(cur,':',emsg)) return false;
                    // skip value generically (string/object/array/number)
                    cur.skipWS();
                    if (cur.peek()=='\"') { std::string dummy; if(!parseString(cur,dummy,emsg)) return false; }
                    else if (cur.peek()=='{') {
                        if (!expect(cur,'{',emsg)) return false;
                        int depth=1;
                        while(!cur.end() && depth>0){ char ch=cur.get(); if(ch=='\"'){ std::string tmp; cur.i--; if(!parseString(cur,tmp,emsg)) return false; } else if(ch=='{') depth++; else if(ch=='}') depth--; }
                        if (depth!=0){ if(emsg)*emsg="Unterminated object"; return false; }
                    } else if (cur.peek()=='[') {
                        if (!expect(cur,'[',emsg)) return false;
                        int depth=1;
                        while(!cur.end() && depth>0){ char ch=cur.get(); if(ch=='\"'){ std::string tmp; cur.i--; if(!parseString(cur,tmp,emsg)) return false; } else if(ch=='[') depth++; else if(ch==']') depth--; }
                        if (depth!=0){ if(emsg)*emsg="Unterminated array"; return false; }
                    } else {
                        double dummyN; if (!parseNumber(cur, dummyN, nullptr)) return false;
                    }
                }
                if (!expect(cur,'}',emsg)) return false;
                break;
            } else {
                // skip value, keep scanning keys
                cur.skipWS();
                if (cur.peek()=='\"') { std::string dummy; if(!parseString(cur,dummy,emsg)) return false; }
                else if (cur.peek()=='{') {
                    if (!expect(cur,'{',emsg)) return false;
                    int depth=1;
                    while(!cur.end() && depth>0){ char ch=cur.get(); if(ch=='\"'){ std::string tmp; cur.i--; if(!parseString(cur,tmp,emsg)) return false; } else if(ch=='{') depth++; else if(ch=='}') depth--; }
                    if (depth!=0){ if(emsg)*emsg="Unterminated object"; return false; }
                } else if (cur.peek()=='[') {
                    if (!expect(cur,'[',emsg)) return false;
                    int depth=1;
                    while(!cur.end() && depth>0){ char ch=cur.get(); if(ch=='\"'){ std::string tmp; cur.i--; if(!parseString(cur,tmp,emsg)) return false; } else if(ch=='[') depth++; else if(ch==']') depth--; }
                    if (depth!=0){ if(emsg)*emsg="Unterminated array"; return false; }
                } else {
                    double dummyN; if (!parseNumber(cur, dummyN, nullptr)) return false;
                }

                cur.skipWS();
                if (!cur.end() && cur.peek()==',') { ++cur.i; }
            }
        }

        if (!foundPipelines) {
            // Not an object with "pipelines": treat it as a single pipeline object
            cur.i = save; // rewind to '{'
            PipelineCfg single;
            if (!parsePipeline(cur, single, emsg)) return false;
            out.pipes.push_back(std::move(single));
        }
    } else {
        if (emsg) *emsg = "Top-level JSON must be object, array, or single pipeline object";
        return false;
    }

    // consume any trailing ws
    cur.skipWS();
    if (!cur.end()) { if (emsg) *emsg = "Trailing characters after JSON"; return false; }

    // put baseDir into pipes
    if (!out.baseDir.empty()) {
        for (auto& pipe: out.pipes) {
            if (pipe.baseDir.empty()) pipe.baseDir = out.baseDir;
        }
    }
    return true;
}