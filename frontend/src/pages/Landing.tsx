import Footer from "../components/common/Footer";
import GlassCard from "../components/common/GlassCard";
import PredictionCard from "../components/dashboard/PredictionCard";
import "./landing.css"

import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
    ArrowRight,
    BrainCircuit,
    ChevronRight,
    MapPinned,
    Satellite,
} from "lucide-react";

import { demoEvents } from "../data/demo/events";

const steps = [
    ["NASA FIRMS", "Satellite observations"],
    ["EVENT FORMATION", "Spatial + temporal evidence"],
    ["EVENT FEATURES", "Event-level intelligence"],
    ["17 LABELING FUNCTIONS", "Weak supervision"],
    ["SNORKEL LABEL MODEL", "Weak labels + probabilities"],
    ["CONFIDENCE + AGREEMENT", "Sample weighting"],
    ["RANDOM FOREST", "Final classifier"],
    ["5-CLASS PREDICTION", "Source intelligence"],
];

const reveal = {
    hidden: {
        opacity: 0,
        y: 34,
    },
    visible: {
        opacity: 1,
        y: 0,
        transition: {
            duration: 0.7,
            ease: [0.22, 1, 0.36, 1],
        },
    },
};

const stagger = {
    hidden: {
        opacity: 0,
    },
    visible: {
        opacity: 1,
        transition: {
            staggerChildren: 0.12,
            delayChildren: 0.1,
        },
    },
};

export default function Landing() {
    const authenticated = Boolean(
        localStorage.getItem("firms_auth_session")
    );

    return (
        <div
            className="landing block w-full min-w-0"
            style={{
                display: "block",
                width: "100%",
                minWidth: 0,
            }}
        >
            {/* Hero Section */}
            <section className="hero">
                <div className="hero-grid" />

                <motion.div
                    className="hero-copy"
                    initial="hidden"
                    animate="visible"
                    variants={stagger}
                >
                    <motion.div className="eyebrow" variants={reveal}>
                        <span />
                        NASA FIRMS · GEOSPATIAL AI · MODEL 2
                    </motion.div>

                    <motion.h1 variants={reveal}>
                        Understand what <em>started</em> the fire.
                    </motion.h1>

                    <motion.p variants={reveal}>
                        An AI-powered fire source classification platform combining NASA
                        FIRMS satellite observations, geospatial evidence, weak
                        supervision, and Random Forest intelligence.
                    </motion.p>

                    <motion.div className="hero-actions" variants={reveal}>
                        {authenticated ? (
                            <>
                                <Link className="btn primary" to="/dashboard">
                                    Open Dashboard
                                    <ArrowRight size={17} />
                                </Link>

                                <Link className="btn glass-btn" to="/model/1">
                                    Compare Models
                                </Link>
                            </>
                        ) : (
                            <>
                                <Link className="btn primary" to="/signup">
                                    Register to explore
                                    <ArrowRight size={17} />
                                </Link>

                                <Link className="btn glass-btn" to="/login">
                                    Sign in
                                </Link>
                            </>
                        )}
                    </motion.div>

                    <motion.div className="signals" variants={reveal}>
                        <span>● NASA FIRMS</span>
                        <span>● AI Classification</span>
                        <span>● Geospatial Intelligence</span>
                    </motion.div>
                </motion.div>

                <motion.div
                    className="hero-orb"
                    initial={{
                        opacity: 0,
                        scale: 0.72,
                        rotate: -10,
                    }}
                    animate={{
                        opacity: 1,
                        scale: 1,
                        rotate: 0,
                    }}
                    transition={{
                        duration: 1.4,
                        ease: [0.16, 1, 0.3, 1],
                        delay: 0.2,
                    }}
                >
                    <div className="earth" />

                    <div className="hotspot h1" />
                    <div className="hotspot h2" />
                    <div className="hotspot h3" />

                    <motion.div
                        className="orbit"
                        animate={{
                            rotate: 360,
                        }}
                        transition={{
                            duration: 24,
                            repeat: Infinity,
                            ease: "linear",
                        }}
                    />
                </motion.div>

                <motion.div
                    className="hero-card"
                    initial={{
                        opacity: 0,
                        x: 32,
                        y: 24,
                    }}
                    animate={{
                        opacity: 1,
                        x: 0,
                        y: 0,
                    }}
                    transition={{
                        duration: 0.9,
                        ease: [0.22, 1, 0.36, 1],
                        delay: 0.6,
                    }}
                >
                    <PredictionCard event={demoEvents[0]} />
                </motion.div>
            </section>

            {/* Idea Section */}
            <motion.section
                className="story section"
                initial="hidden"
                whileInView="visible"
                viewport={{
                    once: true,
                    amount: 0.18,
                }}
                variants={stagger}
            >
                <motion.div className="eyebrow" variants={reveal}>
                    THE IDEA
                </motion.div>

                <motion.h2 variants={reveal}>
                    From satellite detection
                    <br />
                    to <span>source intelligence.</span>
                </motion.h2>

                <motion.p className="lead" variants={reveal}>
                    NASA FIRMS tells us where fire activity is detected. Model 2 adds
                    contextual evidence and machine learning to estimate what kind of
                    activity it may represent.
                </motion.p>

                <motion.div className="three" variants={stagger}>
                    <motion.div variants={reveal}>
                        <GlassCard>
                            <Satellite />

                            <b>DETECT</b>

                            <span>Satellite fire observations</span>
                        </GlassCard>
                    </motion.div>

                    <motion.div variants={reveal}>
                        <GlassCard>
                            <MapPinned />

                            <b>UNDERSTAND</b>

                            <span>Spatial + temporal evidence</span>
                        </GlassCard>
                    </motion.div>

                    <motion.div variants={reveal}>
                        <GlassCard>
                            <BrainCircuit />

                            <b>CLASSIFY</b>

                            <span>Snorkel + Random Forest</span>
                        </GlassCard>
                    </motion.div>
                </motion.div>
            </motion.section>

            {/* Model Teaser Section */}
            <motion.section
                className="model1-teaser section"
                initial="hidden"
                whileInView="visible"
                viewport={{
                    once: true,
                    amount: 0.25,
                }}
                variants={reveal}
            >
                <div className="teaser-inner">
                    <div>
                        <div className="eyebrow">TWO MODEL VIEWS</div>

                        <h2>
                            One intelligence platform.
                            <br />
                            <span>Two model experiences.</span>
                        </h2>

                        <p>
                            Explore Model 1 separately, then dive into Model 2&apos;s
                            Snorkel-Assisted Random Forest pipeline.
                        </p>
                    </div>

                    <Link className="btn primary" to="/model/1">
                        Explore Model 1
                        <ArrowRight size={17} />
                    </Link>
                </div>
            </motion.section>

            {/* Pipeline Section */}
            <motion.section
                className="pipeline section"
                initial="hidden"
                whileInView="visible"
                viewport={{
                    once: true,
                    amount: 0.15,
                }}
                variants={stagger}
            >
                <motion.div className="eyebrow" variants={reveal}>
                    MODEL 2
                </motion.div>

                <motion.h2 variants={reveal}>
                    Inside the intelligence.
                </motion.h2>

                <motion.div className="pipeline-list" variants={stagger}>
                    {steps.map(([title, description], index) => (
                        <motion.div
                            key={title}
                            className="pipe-step"
                            variants={reveal}
                            whileHover={{
                                x: 5,
                            }}
                        >
                            <span>{String(index + 1).padStart(2, "0")}</span>

                            <div>
                                <b>{title}</b>
                                <small>{description}</small>
                            </div>

                            <ChevronRight size={17} />
                        </motion.div>
                    ))}
                </motion.div>
            </motion.section>

            {/* CTA Section */}
            <motion.section
                className="cta section"
                initial="hidden"
                whileInView="visible"
                viewport={{
                    once: true,
                    amount: 0.3,
                }}
                variants={reveal}
            >
                <div>
                    <div className="eyebrow">FIRE SOURCE INTELLIGENCE</div>

                    <h2>
                        Explore the evidence
                        <br />
                        behind every prediction.
                    </h2>

                    <Link className="btn primary" to="/dashboard">
                        Explore Fire Intelligence
                        <ArrowRight size={17} />
                    </Link>
                </div>
            </motion.section>

            {/* Footer */}
            {/* <Footer /> */}
        </div>
    );
}