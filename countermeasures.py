"""Awake-period resistive exercise countermeasures mapped to monitored muscle groups."""

COUNTERMEASURES = {
    "Soleus/Calves": {
        "target": "Plantar flexors (soleus, gastrocnemius)",
        "exercises": (
            ("Bent-knee heel raises", "Loads the soleus, the postural muscle most affected by unloading."),
            ("Straight-knee heel raises", "Loads the gastrocnemius through full ankle range."),
            ("Harnessed treadmill locomotion", "Reproduces repetitive calf loading during gait."),
        ),
    },
    "Quadriceps": {
        "target": "Knee extensors (vastus group, rectus femoris)",
        "exercises": (
            ("Resistive squat", "Primary multi-joint loading of the knee and hip extensors."),
            ("Leg press", "Controlled bilateral quadriceps loading."),
            ("Split squat / lunge", "Unilateral loading to expose side-to-side differences."),
        ),
    },
    "Erector Spinae": {
        "target": "Spinal extensors",
        "exercises": (
            ("Resistive deadlift", "Loads the spinal extensors and posterior chain."),
            ("Back extension", "Isolated endurance loading of the erector spinae."),
            ("Bridge / hip thrust", "Posterior-chain activation with low spinal compression."),
        ),
    },
}
