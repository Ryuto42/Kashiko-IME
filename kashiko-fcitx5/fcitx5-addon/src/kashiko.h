/*
 * Kashiko fcitx5 addon
 */

#ifndef FCITX5_KASHIKO_KASHIKO_H
#define FCITX5_KASHIKO_KASHIKO_H

#include <fcitx/addonfactory.h>
#include <fcitx/addonmanager.h>
#include <fcitx/candidatelist.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputmethodengine.h>
#include <fcitx/instance.h>

// Include the Rust FFI header
#include "../../include/kashiko.h"

namespace fcitx {

class KashikoState;
class KashikoEngine;

// Candidate word class
class KashikoCandidateWord : public CandidateWord {
public:
    KashikoCandidateWord(KashikoEngine* engine, Text text, int index,
                         const std::string& description = "");
    void select(InputContext* inputContext) const override;

private:
    KashikoEngine* engine_;
    int index_;
};

// Candidate list class
class KashikoCandidateList : public CommonCandidateList {
public:
    KashikoCandidateList(KashikoEngine* engine, InputContext* ic);
    void updateCandidates(::KashikoEngine* rustEngine);

private:
    KashikoEngine* engine_;
    InputContext* ic_;
};

// Per-input-context state
class KashikoState : public InputContextProperty {
public:
    KashikoState(KashikoEngine* engine, InputContext* ic);
    ~KashikoState() override;

    void keyEvent(KeyEvent& keyEvent);
    void reset();
    void updateUI();

    ::KashikoEngine* rustEngine() { return rustEngine_; }

private:
    KashikoEngine* engine_;
    InputContext* ic_;
    ::KashikoEngine* rustEngine_{nullptr};
    bool engineInitialized_{false};
};

// Main engine class
class KashikoEngine : public InputMethodEngineV3 {
public:
    KashikoEngine(Instance* instance);
    ~KashikoEngine() override;

    void keyEvent(const InputMethodEntry& entry, KeyEvent& keyEvent) override;
    void reset(const InputMethodEntry& entry, InputContextEvent& event) override;
    void activate(const InputMethodEntry& entry, InputContextEvent& event) override;
    void deactivate(const InputMethodEntry& entry, InputContextEvent& event) override;

    Instance* instance() { return instance_; }

    void selectCandidate(InputContext* ic, int index);

    auto& factory() { return factory_; }

private:
    Instance* instance_;
    FactoryFor<KashikoState> factory_;
};

class KashikoEngineFactory : public AddonFactory {
public:
    AddonInstance* create(AddonManager* manager) override {
        return new KashikoEngine(manager->instance());
    }
};

}  // namespace fcitx

#endif  // FCITX5_KASHIKO_KASHIKO_H
