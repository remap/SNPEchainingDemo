//
// Created by Chiheb Boussema on 22/9/25.
//
// ----- ParseConfig.cpp  -----
#include <string>
#include <unordered_map>
#include <vector>
#include <cctype>
#include <stdexcept>
#include "ParseConfig.hpp"

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
                haveInputs=true;
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

    static bool parsePipeline(Cursor& c, PipelineCfg& cfg, std::string* emsg) {
        // Expects top-level object with "models":[ ... ]
        if (!expect(c,'{',emsg)) return false;

        bool haveModels=false;
        bool haveBaseDir = false;
//        std::string baseDir;
        while (true) {
            c.skipWS();
            if (c.end()) { if (emsg)*emsg="Unterminated top object"; return false; }
            if (c.peek()=='}') { ++c.i; break; }

            std::string key;
            if (!parseString(c,key,emsg)) return false;
            if (!expect(c,':',emsg)) return false;

            if (key=="baseDir") {
                if (!parseString(c,cfg.baseDir,emsg)) continue;
                haveBaseDir=true;
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
            } else {
                // skip unknown field (string / object / array)
                c.skipWS();
                if (c.peek()=='\"') {
                    std::string dummy;
                    if (!parseString(c,dummy,emsg)) return false;
                } else if (c.peek()=='{') {
                    std::unordered_map<std::string,std::string> dummy;
                    if (!parseStringObject(c,dummy,emsg)) return false;
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
                    if (emsg) *emsg = "Unsupported value at top-level key '"+key+"'";
                    return false;
                }
            }

            c.skipWS();
            if (!c.end() && c.peek()==',') { ++c.i; continue; }
            if (!c.end() && c.peek()=='}') { ++c.i; break; }
        }
        if (!haveModels) { if (emsg) *emsg = "Missing 'models' array"; return false; }
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
