/* ============================================================
   ancserAPX — WebGL liquid glass renderer (production)

   Port of the shader bench in
   frontend/static/demos/liquid-glass-switch.html onto the real APX UI.

   Architecture (differs from the demo on purpose):
     - The demo opens one WebGL2 context per control. A production page has
       ~20 glass controls, well past the browser per-page context cap, so this
       module keeps ONE shared WebGL2 context on an offscreen canvas, renders
       each control into a corner of it, and blits the result into that
       control's own lightweight 2D canvas.
     - The scene texture (what the lens refracts) is painted in canvas 2D:
       a shared viewport-sized backdrop (page gradients + panel/chrome fills)
       plus the control's own surface. This mirrors drawBarScene() in the demo.
     - Controls are static when idle: the rAF loop only runs while something is
       animating, then parks. Idle canvases keep their last frame.

   Fallback: if WebGL2 is unavailable or the context is lost, nothing is
   mounted / everything is unmounted and the CSS liquid-glass standard in
   ancserTPX.css remains the visible surface.
   ============================================================ */
(() => {
    'use strict';

    const VERTEX_SHADER_SOURCE = `#version 300 es
                precision highp float;
                out vec2 vUV;

                void main() {
                    vec2 p = vec2(
                        gl_VertexID == 1 ? 2.0 : 0.0,
                        gl_VertexID == 2 ? 2.0 : 0.0
                    );
                    vUV = p;
                    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
                }
            `;

    const FRAGMENT_SHADER_SOURCE = `#version 300 es
                precision highp float;

                uniform sampler2D uScene;
                uniform vec2 uResolution;
                uniform vec2 uTrackCenterPx;
                uniform vec2 uTrackHalfSizePx;
                uniform vec2 uLensCenterPx;
                uniform vec2 uLensHalfSizePx;
                uniform float uInteraction;
                uniform float uStrength;
                uniform float uPixelRatio;
                uniform float uVelocity;
                uniform float uTrackVisibility;
                uniform float uNeutralWeight;
                uniform float uShapeMode;
                uniform float uCornerRadiusPx;
                uniform vec4 uShapeTuning;
                uniform vec4 uInnerTuning;
                uniform vec4 uBlendTuning;
                uniform vec4 uEdgeTuning;
                uniform vec4 uRimTuning;
                uniform vec4 uRoundTuning;
                uniform vec4 uTransitionTuning;

                in vec2 vUV;
                out vec4 fragColor;

                vec2 safeUV(vec2 px) {
                    vec2 halfTexel = vec2(0.5);
                    return clamp(px, halfTexel, uResolution - halfTexel) / uResolution;
                }

                vec4 sceneAt(vec2 px) {
                    return texture(uScene, safeUV(px));
                }

                float capsuleDistance(
                    vec2 point,
                    vec2 halfSize
                ) {
                    float radius = min(halfSize.x, halfSize.y);
                    float axisHalf = max(halfSize.x - radius, 0.0);
                    vec2 axisPoint = vec2(
                        clamp(point.x, -axisHalf, axisHalf),
                        0.0
                    );
                    return length(point - axisPoint) - radius;
                }

                float roundedBoxDistance(
                    vec2 point,
                    vec2 halfSize,
                    float radius
                ) {
                    vec2 q = abs(point) - halfSize + vec2(radius);
                    return length(max(q, 0.0))
                        + min(max(q.x, q.y), 0.0)
                        - radius;
                }

                float lensDistanceAt(vec2 point) {
                    float autoRadius = min(
                        min(uLensHalfSizePx.x, uLensHalfSizePx.y) * 0.31,
                        42.0 * uPixelRatio
                    );
                    float squareRadius = uCornerRadiusPx > 0.0
                        ? min(
                            uCornerRadiusPx,
                            min(uLensHalfSizePx.x, uLensHalfSizePx.y)
                        )
                        : autoRadius;
                    return mix(
                        capsuleDistance(point, uLensHalfSizePx),
                        roundedBoxDistance(
                            point,
                            uLensHalfSizePx,
                            squareRadius
                        ),
                        uShapeMode
                    );
                }

                vec2 lensDistanceNormal(vec2 point) {
                    float eps = max(0.75, uPixelRatio);
                    vec2 gradient = vec2(
                        lensDistanceAt(point + vec2(eps, 0.0))
                            - lensDistanceAt(point - vec2(eps, 0.0)),
                        lensDistanceAt(point + vec2(0.0, eps))
                            - lensDistanceAt(point - vec2(0.0, eps))
                    );
                    return length(gradient) > 0.0001
                        ? normalize(gradient)
                        : vec2(0.0, 1.0);
                }

                float trackSignedDistanceAt(vec2 px) {
                    vec2 d = px - uTrackCenterPx;
                    float radiusPx = uTrackHalfSizePx.y;
                    float axisHalfPx = max(uTrackHalfSizePx.x - radiusPx, 0.0);
                    vec2 axisPoint = vec2(clamp(d.x, -axisHalfPx, axisHalfPx), 0.0);
                    return length(d - axisPoint) - radiusPx;
                }

                float trackMaskAt(vec2 px) {
                    float signedDistance = trackSignedDistanceAt(px);
                    float aa = max(fwidth(signedDistance), 0.75);
                    return 1.0 - smoothstep(-aa, aa, signedDistance);
                }

                void main() {
                    vec2 fragPx = vUV * uResolution;
                    vec4 base = sceneAt(fragPx);
                    float trackMask = trackMaskAt(fragPx);
                    float trackSignedDistance =
                        trackSignedDistanceAt(fragPx);
                    vec2 trackDelta = fragPx - uTrackCenterPx;
                    float trackRadiusPx = uTrackHalfSizePx.y;
                    float trackAxisHalfPx = max(
                        uTrackHalfSizePx.x - trackRadiusPx,
                        0.0
                    );
                    vec2 trackAxisPoint = vec2(
                        clamp(
                            trackDelta.x,
                            -trackAxisHalfPx,
                            trackAxisHalfPx
                        ),
                        0.0
                    );
                    vec2 trackRadial = trackDelta - trackAxisPoint;
                    vec2 trackNormal = length(trackRadial) > 0.0001
                        ? normalize(trackRadial)
                        : vec2(0.0, 1.0);
                    float trackEdgeWidthPx = 6.0 * uPixelRatio;
                    float neutralTrackBand =
                        smoothstep(
                            -trackEdgeWidthPx,
                            -0.45 * uPixelRatio,
                            trackSignedDistance
                        )
                        * (1.0 - smoothstep(
                            0.0,
                            max(fwidth(trackSignedDistance), 0.75),
                            trackSignedDistance
                        ))
                        * trackMask
                        * uTrackVisibility;
                    vec4 neutralTrackRefraction = sceneAt(
                        fragPx
                            - trackNormal
                            * mix(1.15, 2.05, uStrength)
                            * uPixelRatio
                    );
                    base = mix(
                        base,
                        neutralTrackRefraction,
                        neutralTrackBand * mix(0.34, 0.54, uStrength)
                    );

                    vec2 d = fragPx - uLensCenterPx;
                    float lensRadiusPx = min(
                        uLensHalfSizePx.x,
                        uLensHalfSizePx.y
                    );
                    float lensAxisHalfPx = max(
                        uLensHalfSizePx.x - lensRadiusPx,
                        0.0
                    );
                    vec2 lensAxisPoint = vec2(
                        clamp(d.x, -lensAxisHalfPx, lensAxisHalfPx),
                        0.0
                    );
                    vec2 lensRadial = d - lensAxisPoint;
                    float lensRadialLength = length(lensRadial);
                    float glassActivation = smoothstep(
                        0.08,
                        0.36,
                        uInteraction
                    );
                    float lensSignedDistance = lensDistanceAt(d);
                    float lensAA = max(
                        fwidth(lensSignedDistance) * 1.15,
                        0.75
                    );
                    float lensMask = 1.0 - smoothstep(
                        -lensAA,
                        lensAA,
                        lensSignedDistance
                    );
                    float outputMask = max(
                        trackMask * uTrackVisibility,
                        lensMask
                    );

                    if (outputMask <= 0.001) {
                        fragColor = vec4(0.0);
                        return;
                    }

                    vec2 capsuleNormal = lensRadialLength > 0.0001
                        ? lensRadial / lensRadialLength
                        : vec2(0.0, 1.0);
                    vec2 normal2D = normalize(mix(
                        capsuleNormal,
                        lensDistanceNormal(d),
                        uShapeMode
                    ));
                    vec2 tangent = vec2(-normal2D.y, normal2D.x);
                    float edgeWidthPx =
                        mix(uShapeTuning.x, uShapeTuning.y, uInteraction)
                            * uPixelRatio;
                    float floatingLensWeight = 1.0 - uTrackVisibility;
                    float transitionWidthPx = max(
                        edgeWidthPx
                            * mix(
                                uInnerTuning.x,
                                uInnerTuning.y,
                                floatingLensWeight
                            ),
                        uTransitionTuning.x
                            * uPixelRatio
                            * mix(0.86, 1.18, floatingLensWeight)
                    );
                    float transitionDecay = max(uTransitionTuning.y, 0.05);
                    float rgbBandWidthPx = max(
                        edgeWidthPx,
                        uTransitionTuning.z * uPixelRatio
                    );
                    float edgeTaperPower = max(uTransitionTuning.w, 0.05);
                    float normalizedLensDepth = clamp(
                        -lensSignedDistance / max(lensRadiusPx, 1.0),
                        0.0,
                        1.0
                    );
                    float interiorContentBand =
                        smoothstep(0.14, 0.52, normalizedLensDepth)
                        * lensMask
                        * glassActivation;
                    float contentSampleScale =
                        1.0
                        + (
                            uShapeTuning.z
                            + floatingLensWeight * uShapeTuning.w
                        )
                            * uInteraction
                            * mix(0.72, 1.0, uStrength);
                    vec2 compressedContentPx =
                        uLensCenterPx + d * contentSampleScale;
                    vec4 compressedContent = sceneAt(compressedContentPx);
                    float transitionLinear = smoothstep(
                        -transitionWidthPx,
                        -0.54 * uPixelRatio,
                        lensSignedDistance
                    );
                    float transitionProgress = clamp(
                        (1.0 - exp(-transitionLinear * transitionDecay))
                            / max(1.0 - exp(-transitionDecay), 0.001),
                        0.0,
                        1.0
                    );
                    float innerEdgeProgress =
                        transitionProgress * lensMask * glassActivation;
                    float edgeFoldCurve = pow(
                        transitionProgress,
                        mix(1.18, 0.92, floatingLensWeight)
                    ) * lensMask * glassActivation;
                    float sideRoundPull =
                        mix(
                            uRoundTuning.x,
                            uRoundTuning.y,
                            pow(abs(normal2D.x), 0.62)
                        );
                    float innerRefractionPullPx =
                        (
                            uInnerTuning.z * mix(0.72, 1.0, uStrength)
                            + floatingLensWeight * uInnerTuning.w
                        )
                        * uPixelRatio
                        * uInteraction
                        * sideRoundPull;
                    vec2 innerWarpedContentPx =
                        compressedContentPx
                        - normal2D
                            * innerRefractionPullPx
                            * edgeFoldCurve
                        + tangent
                            * dot(d, tangent)
                            / max(lensRadiusPx, 1.0)
                            * innerRefractionPullPx
                            * 0.36
                            * edgeFoldCurve;
                    vec4 innerWarpedContent = sceneAt(innerWarpedContentPx);
                    base = mix(
                        base,
                        compressedContent,
                        max(
                            interiorContentBand,
                            lensMask * smoothstep(0.02, 0.14, uInteraction)
                        )
                    );
                    base = mix(
                        base,
                        innerWarpedContent,
                        innerEdgeProgress
                            * mix(uBlendTuning.x, uBlendTuning.y, uInteraction)
                    );

                    float edgeProgress = smoothstep(
                        -edgeWidthPx,
                        0.0,
                        lensSignedDistance
                    );
                    float edgeBand =
                        smoothstep(
                            -edgeWidthPx,
                            -0.65 * uPixelRatio,
                            lensSignedDistance
                        )
                        * (1.0 - smoothstep(
                            0.0,
                            lensAA,
                            lensSignedDistance
                        ))
                        * lensMask
                        * glassActivation;
                    float rgbSpreadProgress = smoothstep(
                        -rgbBandWidthPx,
                        0.0,
                        lensSignedDistance
                    );
                    float rgbHaloBand =
                        smoothstep(
                            -rgbBandWidthPx,
                            -edgeWidthPx * 0.45,
                            lensSignedDistance
                        )
                        * (1.0 - smoothstep(
                            0.0,
                            lensAA,
                            lensSignedDistance
                        ))
                        * lensMask
                        * glassActivation;

                    if (max(edgeBand, rgbHaloBand) <= 0.001) {
                        fragColor = vec4(base.rgb * outputMask, outputMask);
                        return;
                    }

                    float edgeInnerTaper = pow(
                        clamp(1.0 - edgeProgress * 0.92, 0.0, 1.0),
                        edgeTaperPower
                    );
                    float edgeOpticBand = edgeBand * mix(
                        1.0,
                        edgeInnerTaper,
                        0.62
                    );
                    float displacementPx =
                        mix(5.0, 9.0, uStrength)
                        * uPixelRatio
                        * mix(0.82, 1.16, uInteraction);
                    vec2 trackToLensRatio = clamp(
                        uTrackHalfSizePx / uLensHalfSizePx,
                        vec2(0.05),
                        vec2(1.0)
                    );
                    vec2 boundaryMappedPx =
                        uLensCenterPx + d * trackToLensRatio;
                    vec2 refractedPx = mix(
                        fragPx,
                        boundaryMappedPx,
                        edgeProgress
                            * uInteraction
                            * mix(0.64, 0.88, uStrength)
                    ) - normal2D
                        * displacementPx
                        * pow(edgeProgress, 1.35)
                        + vec2(
                            -uVelocity
                                * 3.2
                                * uPixelRatio
                                * edgeProgress,
                            0.0
                        );

                    float mirrorRadiusPx = lensRadiusPx;
                    float mirroredDepthPx = mirrorRadiusPx
                        * mix(0.52, 0.06, edgeProgress);
                    vec2 reflectedPx =
                        uLensCenterPx
                        - normal2D * mirroredDepthPx
                        + tangent * dot(d, tangent) * 0.30;

                    float blurRadiusPx =
                        uEdgeTuning.x * uPixelRatio;
                    vec4 refracted =
                        sceneAt(refractedPx) * 0.40
                        + sceneAt(
                            refractedPx + tangent * blurRadiusPx
                        ) * 0.15
                        + sceneAt(
                            refractedPx - tangent * blurRadiusPx
                        ) * 0.15
                        + sceneAt(
                            refractedPx + normal2D * blurRadiusPx
                        ) * 0.15
                        + sceneAt(
                            refractedPx - normal2D * blurRadiusPx
                        ) * 0.15;
                    float edgeCrispness = smoothstep(
                        uEdgeTuning.y,
                        uEdgeTuning.z,
                        edgeProgress
                    );
                    refracted = mix(
                        refracted,
                        sceneAt(refractedPx),
                        edgeCrispness * 0.72
                    );
                    float chromaticShiftPx =
                        uEdgeTuning.w
                        * mix(0.72, 1.0, uStrength)
                        * uPixelRatio;
                    float rgbHaloProgress = pow(
                        rgbSpreadProgress,
                        max(0.35, edgeTaperPower * 0.55)
                    );
                    vec2 chromaticBasePx = mix(
                        compressedContentPx,
                        innerWarpedContentPx,
                        clamp(
                            innerEdgeProgress
                                + rgbHaloBand * rgbHaloProgress * 0.72,
                            0.0,
                            1.0
                        )
                    );
                    vec2 dispersionRay =
                        chromaticBasePx - compressedContentPx;
                    vec2 dispersionDirection =
                        length(dispersionRay) > 0.01
                            ? normalize(dispersionRay)
                            : -normal2D;
                    vec4 redChannel = sceneAt(
                        chromaticBasePx
                            + dispersionDirection
                            * chromaticShiftPx
                            * 0.82
                    );
                    vec4 greenChannel = sceneAt(chromaticBasePx);
                    vec4 blueChannel = sceneAt(
                        chromaticBasePx
                            - dispersionDirection
                            * chromaticShiftPx
                    );
                    vec3 dispersedRGB = vec3(
                        redChannel.r,
                        greenChannel.g,
                        blueChannel.b
                    );
                    float chromaticBand =
                        max(
                            smoothstep(0.30, 0.94, edgeProgress),
                            rgbHaloBand * rgbHaloProgress
                        );
                    base.rgb = mix(
                        base.rgb,
                        dispersedRGB,
                        rgbHaloBand
                            * rgbHaloProgress
                            * mix(uRoundTuning.z, uRoundTuning.w, uInteraction)
                            * 0.78
                    );
                    vec4 reflected = sceneAt(reflectedPx);
                    float reflectionMix =
                        smoothstep(
                            -5.0 * uPixelRatio,
                            0.0,
                            lensSignedDistance
                        )
                        * mix(0.14, 0.25, uStrength)
                        * mix(0.72, 1.0, uInteraction)
                        * mix(0.58, 1.0, uNeutralWeight);
                    vec4 edgePixels = mix(refracted, reflected, reflectionMix);
                    edgePixels.rgb = mix(
                        edgePixels.rgb,
                        dispersedRGB,
                        chromaticBand
                            * mix(uRoundTuning.z, uRoundTuning.w, uInteraction)
                            * 1.18
                            * (1.0 - reflectionMix * 0.45)
                    );
                    float rimFoldBand =
                        edgeBand
                        * smoothstep(uBlendTuning.z, 0.76, edgeProgress);
                    float rimFoldProgress = pow(
                        smoothstep(0.04, 1.0, edgeProgress),
                        0.92
                    );
                    vec2 centerFoldPx = mix(
                        innerWarpedContentPx,
                        reflectedPx,
                        rimFoldProgress
                    );
                    vec4 foldedContent = sceneAt(centerFoldPx);
                    edgePixels.rgb = mix(
                        edgePixels.rgb,
                        foldedContent.rgb,
                        rimFoldBand * uBlendTuning.w * uInteraction
                    );
                    float clearRimBand =
                        smoothstep(
                            -uRimTuning.x * uPixelRatio,
                            -0.38 * uPixelRatio,
                            lensSignedDistance
                        )
                        * (1.0 - smoothstep(
                            0.04 * uPixelRatio,
                            lensAA,
                            lensSignedDistance
                        ))
                        * lensMask
                        * glassActivation;
                    float clearRimLuma = dot(
                        edgePixels.rgb,
                        vec3(0.2126, 0.7152, 0.0722)
                    );
                    float rimRoundness =
                        smoothstep(0.30, 1.0, edgeProgress)
                        * mix(
                            0.72,
                            1.0,
                            pow(abs(normal2D.x), 0.72)
                        );
                    vec3 clearRimTone = clamp(
                        edgePixels.rgb
                            * mix(
                                0.78,
                                0.54,
                                smoothstep(0.40, 0.72, clearRimLuma)
                                    * rimRoundness
                            ),
                        0.0,
                        1.0
                    );
                    edgePixels.rgb = mix(
                        edgePixels.rgb,
                        clearRimTone,
                        clearRimBand * uRimTuning.y * uInteraction
                    );
                    vec4 glassPixels = mix(
                        base,
                        edgePixels,
                        max(edgeOpticBand, clearRimBand * 0.96)
                    );

                    float shoulderWidthPx =
                        uRimTuning.z * uPixelRatio;
                    float shoulderBand =
                        smoothstep(
                            -shoulderWidthPx,
                            -1.1 * uPixelRatio,
                            lensSignedDistance
                        )
                        * (1.0 - smoothstep(
                            -0.15 * uPixelRatio,
                            lensAA,
                            lensSignedDistance
                        ))
                        * lensMask
                        * glassActivation;
                    float directionalLight = clamp(
                        0.52
                            + dot(
                                normal2D,
                                normalize(vec2(-0.62, -0.78))
                            ) * 0.48,
                        0.0,
                        1.0
                    );
                    float fresnel = pow(edgeProgress, 6.0);
                    float rimLuma = dot(
                        glassPixels.rgb,
                        vec3(0.2126, 0.7152, 0.0722)
                    );
                    vec3 shoulderTone = clamp(
                        glassPixels.rgb * mix(
                            1.18,
                            0.82,
                            smoothstep(0.38, 0.64, rimLuma)
                        ),
                        0.0,
                        1.0
                    );
                    glassPixels.rgb = mix(
                        glassPixels.rgb,
                        shoulderTone,
                        shoulderBand
                            * (0.12 + directionalLight * 0.22)
                            * (0.72 + fresnel * 0.28)
                            * mix(0.28, 1.0, uNeutralWeight)
                    );

                    float coreWidthPx =
                        uRimTuning.w * uPixelRatio;
                    float coreBand =
                        smoothstep(
                            -coreWidthPx,
                            -0.28 * uPixelRatio,
                            lensSignedDistance
                        )
                        * (1.0 - smoothstep(
                            0.0,
                            lensAA,
                            lensSignedDistance
                        ))
                        * lensMask
                        * glassActivation;
                    vec3 coreTone = clamp(
                        glassPixels.rgb * mix(
                            1.30,
                            0.70,
                            smoothstep(0.40, 0.66, rimLuma)
                        ),
                        0.0,
                        1.0
                    );
                    glassPixels.rgb = mix(
                        glassPixels.rgb,
                        coreTone,
                        coreBand
                            * mix(0.56, 0.70, uInteraction)
                            * mix(0.24, 1.0, uNeutralWeight)
                    );

                    float topShadowBand =
                        coreBand
                        * smoothstep(0.15, 0.92, -normal2D.y)
                        * uInteraction;
                    float bottomLightBand =
                        coreBand
                        * smoothstep(0.15, 0.92, normal2D.y)
                        * uInteraction;
                    glassPixels.rgb *= 1.0 - topShadowBand * 0.22;
                    vec3 lowerLightTone = clamp(
                        glassPixels.rgb * 1.28 + vec3(0.035),
                        0.0,
                        1.0
                    );
                    glassPixels.rgb = mix(
                        glassPixels.rgb,
                        lowerLightTone,
                        bottomLightBand * 0.72
                    );

                    fragColor = vec4(
                        glassPixels.rgb * outputMask,
                        outputMask
                    );
                }
            `;

    // Calibrated defaults from the demo tuning bench (its reset values).
    const TUNING = {
        edgeIdle: 3,
        edgeActive: 24,
        centerShrink: 0.14,
        floatingShrink: 0.22,
        innerSpanTrack: 1.05,
        innerSpanFloating: 4.6,
        transitionSpread: 42,
        transitionDecay: 0.2,
        innerMixMin: 0,
        innerMixMax: 1,
        sideRoundMin: 0.2,
        sideRoundMax: 2.6,
        rimFoldStart: 0.20,
        rimFoldMix: 0.20,
        rimWidth: 1.3,
        rimStrength: 4,
        chromaShift: 5,
        rgbBandWidth: 36,
        chromaMin: 0,
        chromaMax: 1.20,
        edgeTaper: 6,
        edgeCrispStart: 0,
        edgeCrispEnd: 1,
        shoulderWidth: 2.4,
        coreWidth: 9.5,
    };

    // Demo strength slider default (52 / 100).
    const STRENGTH = 0.52;

    // The demo's md switch is 58px tall; every px tuning value above is
    // expressed against that. Production controls are smaller, so each node
    // scales the px tuning by its own height (see APX_GLASS_STANDARD.md).
    const REFERENCE_HEIGHT_PX = 58;

    // Mirrors the body and .main::before background stacks in ancserTPX.css,
    // per theme. Keep in sync when the page background changes — this is what
    // the lens refracts.
    const BACKDROP = {
        dark: {
            base: '#05070b',
            gradient: {
                angle: 135,
                stops: [[0, '#05070b'], [0.48, '#090b12'], [1, '#050509']],
            },
            page: [
                { x: 0.16, y: 0.22, stop: 0.26, color: [255, 167, 38, 0.16] },
                { x: 0.72, y: 0.18, stop: 0.28, color: [100, 220, 255, 0.14] },
                { x: 0.76, y: 0.76, stop: 0.26, color: [0, 229, 160, 0.12] },
            ],
            main: [
                { x: 0.30, y: 0.36, stop: 0.30, color: [255, 64, 96, 0.10] },
                { x: 0.83, y: 0.46, stop: 0.32, color: [100, 220, 255, 0.10] },
            ],
        },
        light: {
            base: '#f3f3ef',
            gradient: {
                angle: 135,
                stops: [[0, '#f6f6f2'], [0.48, '#ecece6'], [1, '#f2f2ee']],
            },
            page: [
                { x: 0.16, y: 0.22, stop: 0.26, color: [255, 167, 38, 0.20] },
                { x: 0.72, y: 0.18, stop: 0.28, color: [100, 200, 255, 0.20] },
                { x: 0.76, y: 0.76, stop: 0.26, color: [0, 200, 150, 0.18] },
            ],
            main: [
                { x: 0.30, y: 0.36, stop: 0.30, color: [204, 31, 60, 0.09] },
                { x: 0.83, y: 0.46, stop: 0.32, color: [10, 124, 168, 0.09] },
            ],
        },
    };

    // Scene fills painted under the lens, per theme.
    const SCENE = {
        dark: {
            surface: 'rgba(11, 14, 21, 0.58)',
            track: 'rgba(24, 28, 35, 0.88)',
            trackStroke: 'rgba(234, 240, 255, 0.18)',
            pill: 'rgb(68, 74, 86)',
            pillOn: 'rgb(0, 176, 122)',
            pillStroke: 'rgba(255, 255, 255, 0.26)',
            thumbOff: [242, 246, 252],
            thumbOn: [255, 255, 255],
            thumbRim: 'rgba(0, 0, 0, 0.42)',
            navQuiet: [34, 38, 44],
        },
        light: {
            surface: 'rgba(255, 255, 255, 0.62)',
            track: 'rgba(203, 206, 211, 0.92)',
            trackStroke: 'rgba(18, 28, 44, 0.20)',
            pill: 'rgb(128, 136, 152)',
            pillOn: 'rgb(0, 132, 90)',
            pillStroke: 'rgba(18, 28, 44, 0.26)',
            thumbOff: [255, 255, 255],
            thumbOn: [255, 255, 255],
            thumbRim: 'rgba(18, 28, 44, 0.45)',
            navQuiet: [255, 255, 255],
        },
    };
    // Opaque-ish chrome painted into the backdrop so controls sitting on a
    // panel refract the panel, not the raw page gradient.
    const BACKDROP_SURFACES = [
        '.header',
        '.sidebar',
        '.bottom-panel',
        '.panel.apx-glass-container',
        '.conn-panel.open',
    ];

    const state = {
        nodes: [],
        raf: 0,
        frames: 0,
        renders: 0,
        lastTime: 0,
        dpr: 1,
        gl: null,
        glCanvas: null,
        program: null,
        texture: null,
        uniforms: null,
        textureWidth: 0,
        textureHeight: 0,
        available: false,
        disabled: false,
        backdrop: null,
        backdropCtx: null,
        backdropWidth: 0,
        backdropHeight: 0,
        backdropDirty: true,
        reduceMotion: false,
        keyboardNav: false,
        theme: {},
        mode: 'dark',
        scene: SCENE.dark,
        backdropLayers: BACKDROP.dark,
        scrollTimer: 0,
        refreshTimer: 0,
        applying: false,
        observer: null,
        contextFailures: 0,
    };

    /* ── small helpers ───────────────────────────────────────────────────── */

    const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

    function getDpr() {
        const coarse = window.matchMedia('(pointer: coarse)').matches;
        return Math.min(window.devicePixelRatio || 1, coarse ? 1.25 : 1.5);
    }

    function rgba(color, alphaScale = 1) {
        const [r, g, b, a = 1] = color;
        return `rgba(${r}, ${g}, ${b}, ${(a * alphaScale).toFixed(4)})`;
    }

    function cssVar(name, fallback) {
        const value = getComputedStyle(document.documentElement)
            .getPropertyValue(name)
            .trim();
        return value || fallback;
    }

    function readTheme() {
        const mode = document.documentElement.dataset.theme === 'light'
            ? 'light'
            : 'dark';
        state.mode = mode;
        state.scene = SCENE[mode];
        state.backdropLayers = BACKDROP[mode];
        state.theme = {
            green: cssVar('--green', '#00e5a0'),
            amber: cssVar('--amber', '#ffa726'),
            cyan: cssVar('--cyan', '#64dcff'),
            white: cssVar('--white', '#eaf0ff'),
            text2: cssVar('--text2', '#556178'),
        };
    }

    function roundedRectPath(ctx, x, y, width, height, radius) {
        const r = Math.max(0, Math.min(radius, width / 2, height / 2));
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + width - r, y);
        ctx.arcTo(x + width, y, x + width, y + r, r);
        ctx.lineTo(x + width, y + height - r);
        ctx.arcTo(x + width, y + height, x + width - r, y + height, r);
        ctx.lineTo(x + r, y + height);
        ctx.arcTo(x, y + height, x, y + height - r, r);
        ctx.lineTo(x, y + r);
        ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
    }

    function parseRadius(value, box) {
        if (!value) return 0;
        const first = String(value).split(' ')[0];
        if (first.endsWith('%')) {
            return (parseFloat(first) / 100) * Math.min(box.width, box.height);
        }
        const px = parseFloat(first);
        return Number.isFinite(px) ? px : 0;
    }

    /* ── shared WebGL renderer ───────────────────────────────────────────── */

    function compileShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const message = gl.getShaderInfoLog(shader) || 'shader compile failed';
            gl.deleteShader(shader);
            throw new Error(message);
        }
        return shader;
    }

    function createProgram(gl) {
        const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE);
        const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);
        const program = gl.createProgram();
        gl.attachShader(program, vertex);
        gl.attachShader(program, fragment);
        gl.linkProgram(program);
        gl.deleteShader(vertex);
        gl.deleteShader(fragment);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const message = gl.getProgramInfoLog(program) || 'program link failed';
            gl.deleteProgram(program);
            throw new Error(message);
        }
        return program;
    }

    function initRenderer() {
        if (state.available || state.disabled) return state.available;
        try {
            const canvas = document.createElement('canvas');
            canvas.width = 1;
            canvas.height = 1;
            const gl = canvas.getContext('webgl2', {
                alpha: true,
                antialias: false,
                depth: false,
                stencil: false,
                premultipliedAlpha: true,
                preserveDrawingBuffer: true,
                powerPreference: 'high-performance',
            });
            if (!gl) throw new Error('WebGL2 unavailable');

            const program = createProgram(gl);
            const texture = gl.createTexture();
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

            gl.useProgram(program);
            gl.uniform1i(gl.getUniformLocation(program, 'uScene'), 0);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
            gl.clearColor(0, 0, 0, 0);

            canvas.addEventListener('webglcontextlost', (event) => {
                event.preventDefault();
                state.contextFailures += 1;
                teardown();
            });

            state.glCanvas = canvas;
            state.gl = gl;
            state.program = program;
            state.texture = texture;
            state.textureWidth = 0;
            state.textureHeight = 0;
            state.uniforms = {
                resolution: gl.getUniformLocation(program, 'uResolution'),
                trackCenter: gl.getUniformLocation(program, 'uTrackCenterPx'),
                trackHalfSize: gl.getUniformLocation(program, 'uTrackHalfSizePx'),
                lensCenter: gl.getUniformLocation(program, 'uLensCenterPx'),
                lensHalfSize: gl.getUniformLocation(program, 'uLensHalfSizePx'),
                interaction: gl.getUniformLocation(program, 'uInteraction'),
                strength: gl.getUniformLocation(program, 'uStrength'),
                pixelRatio: gl.getUniformLocation(program, 'uPixelRatio'),
                velocity: gl.getUniformLocation(program, 'uVelocity'),
                trackVisibility: gl.getUniformLocation(program, 'uTrackVisibility'),
                neutralWeight: gl.getUniformLocation(program, 'uNeutralWeight'),
                shapeMode: gl.getUniformLocation(program, 'uShapeMode'),
                cornerRadius: gl.getUniformLocation(program, 'uCornerRadiusPx'),
                shapeTuning: gl.getUniformLocation(program, 'uShapeTuning'),
                innerTuning: gl.getUniformLocation(program, 'uInnerTuning'),
                blendTuning: gl.getUniformLocation(program, 'uBlendTuning'),
                edgeTuning: gl.getUniformLocation(program, 'uEdgeTuning'),
                rimTuning: gl.getUniformLocation(program, 'uRimTuning'),
                roundTuning: gl.getUniformLocation(program, 'uRoundTuning'),
                transitionTuning: gl.getUniformLocation(program, 'uTransitionTuning'),
            };
            state.available = true;
            document.documentElement.dataset.apxGlassRenderer = 'webgl';
            return true;
        } catch (error) {
            state.available = false;
            state.disabled = true;
            state.contextFailures += 1;
            document.documentElement.dataset.apxGlassRenderer = 'css';
            return false;
        }
    }

    function ensureRenderTarget(width, height) {
        const canvas = state.glCanvas;
        const nextWidth = Math.max(canvas.width, width);
        const nextHeight = Math.max(canvas.height, height);
        if (nextWidth !== canvas.width || nextHeight !== canvas.height) {
            canvas.width = nextWidth;
            canvas.height = nextHeight;
        }
    }

    /* ── shared backdrop (what sits behind the glass) ────────────────────── */

    function paintGlowLayer(ctx, layer, box) {
        const centerX = box.left + box.width * layer.x;
        const centerY = box.top + box.height * layer.y;
        const corners = [
            Math.hypot(centerX - box.left, centerY - box.top),
            Math.hypot(box.left + box.width - centerX, centerY - box.top),
            Math.hypot(centerX - box.left, box.top + box.height - centerY),
            Math.hypot(box.left + box.width - centerX, box.top + box.height - centerY),
        ];
        const radius = Math.max(1, Math.max(...corners) * layer.stop);
        const gradient = ctx.createRadialGradient(
            centerX, centerY, 0, centerX, centerY, radius
        );
        gradient.addColorStop(0, rgba(layer.color));
        gradient.addColorStop(1, rgba(layer.color, 0));
        ctx.fillStyle = gradient;
        ctx.fillRect(box.left, box.top, box.width, box.height);
    }

    function paintBackdrop() {
        const dpr = state.dpr;
        const width = Math.max(1, window.innerWidth);
        const height = Math.max(1, window.innerHeight);
        if (!state.backdrop) {
            state.backdrop = document.createElement('canvas');
            state.backdropCtx = state.backdrop.getContext('2d', { alpha: false });
        }
        const bitmapWidth = Math.round(width * dpr);
        const bitmapHeight = Math.round(height * dpr);
        if (
            state.backdrop.width !== bitmapWidth
            || state.backdrop.height !== bitmapHeight
        ) {
            state.backdrop.width = bitmapWidth;
            state.backdrop.height = bitmapHeight;
        }
        state.backdropWidth = width;
        state.backdropHeight = height;

        const ctx = state.backdropCtx;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const viewport = { left: 0, top: 0, width, height };
        const layers = state.backdropLayers;
        const angle = (layers.gradient.angle - 90) * (Math.PI / 180);
        const span = Math.abs(width * Math.cos(angle)) + Math.abs(height * Math.sin(angle));
        const gradient = ctx.createLinearGradient(
            width / 2 - (Math.cos(angle) * span) / 2,
            height / 2 - (Math.sin(angle) * span) / 2,
            width / 2 + (Math.cos(angle) * span) / 2,
            height / 2 + (Math.sin(angle) * span) / 2
        );
        layers.gradient.stops.forEach(([offset, color]) => {
            gradient.addColorStop(offset, color);
        });
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        layers.page.forEach((layer) => paintGlowLayer(ctx, layer, viewport));

        const main = document.querySelector('.main');
        if (main) {
            const rect = main.getBoundingClientRect();
            const box = {
                left: rect.left,
                top: rect.top,
                width: rect.width,
                height: rect.height,
            };
            ctx.save();
            ctx.beginPath();
            ctx.rect(box.left, box.top, box.width, box.height);
            ctx.clip();
            layers.main.forEach((layer) => paintGlowLayer(ctx, layer, box));
            ctx.restore();
        }

        BACKDROP_SURFACES.forEach((selector) => {
            document.querySelectorAll(selector).forEach((element) => {
                const style = getComputedStyle(element);
                if (style.display === 'none' || style.visibility === 'hidden') return;
                const color = style.backgroundColor;
                if (!color || color === 'transparent' || color === 'rgba(0, 0, 0, 0)') return;
                const rect = element.getBoundingClientRect();
                if (rect.width < 1 || rect.height < 1) return;
                if (
                    rect.right < 0 || rect.bottom < 0
                    || rect.left > width || rect.top > height
                ) return;
                roundedRectPath(
                    ctx,
                    rect.left,
                    rect.top,
                    rect.width,
                    rect.height,
                    parseRadius(style.borderRadius, rect)
                );
                ctx.fillStyle = color;
                ctx.fill();
            });
        });

        state.backdropDirty = false;
    }

    function drawBackdropInto(ctx, node) {
        const rect = node.canvasRect;
        ctx.fillStyle = state.backdropLayers.base;
        ctx.fillRect(0, 0, node.canvasWidth, node.canvasHeight);
        if (!state.backdrop || !state.backdrop.width) return;
        ctx.drawImage(
            state.backdrop,
            0,
            0,
            state.backdrop.width,
            state.backdrop.height,
            -rect.left,
            -rect.top,
            state.backdropWidth,
            state.backdropHeight
        );
    }

    /* ── node lifecycle ──────────────────────────────────────────────────── */

    // A geometry may describe one lens with flat fields or several via
    // `lenses`; the renderer and the motion loop always see a list.
    function lensesOf(geometry) {
        if (geometry.lenses) return geometry.lenses;
        return [{
            x: geometry.lensX,
            y: geometry.lensY,
            halfWidth: geometry.halfWidth,
            halfHeight: geometry.halfHeight,
            grow: geometry.grow,
        }];
    }

    function createNode(config) {
        const canvas = document.createElement('canvas');
        canvas.className = 'apx-glass-canvas';
        canvas.setAttribute('aria-hidden', 'true');
        const context = canvas.getContext('2d', { alpha: true });
        const scene = document.createElement('canvas');
        const sceneContext = scene.getContext('2d', { alpha: true });

        const node = {
            type: config.type,
            element: config.element,
            host: config.host,
            canvas,
            context,
            scene,
            sceneContext,
            padX: config.padX,
            padY: config.padY,
            shape: config.shape || 'capsule',
            trackVisibility: config.trackVisibility,
            neutralWeight: config.neutralWeight,
            idleInteraction: config.idleInteraction,
            springMotion: Boolean(config.springMotion),
            hoverActivates: config.hoverActivates !== false,
            paintSurface: config.paintSurface,
            syncDom: config.syncDom || null,
            geometry: config.geometry,
            progress: typeof config.progress === 'function' ? config.progress : () => 0,
            segments: config.segments || null,
            tuning: config.tuning ? { ...TUNING, ...config.tuning } : TUNING,
            width: 0,
            height: 0,
            dpr: state.dpr,
            sizeScale: 1,
            canvasWidth: 0,
            canvasHeight: 0,
            canvasRect: { left: 0, top: 0 },
            x: 0,
            targetX: 0,
            positions: [],
            targets: [],
            sizes: [],
            sizeTargets: [],
            velocityUnit: 0,
            velocityPx: 0,
            activity: config.idleInteraction,
            activityVelocity: 0,
            interaction: config.idleInteraction,
            hover: false,
            pressed: false,
            focused: false,
            dragging: false,
            visible: true,
            dirty: true,
            mounted: false,
        };

        node.host.appendChild(canvas);
        config.element.dataset.apxRenderer = 'webgl';
        bindNodeEvents(node);
        state.nodes.push(node);
        node.mounted = true;
        layoutNode(node);
        node.positions = node.targets.slice();
        node.x = node.targetX;
        if (node.syncDom) node.syncDom(node);
        return node;
    }

    function unmountNode(node) {
        node.mounted = false;
        if (node.canvas.parentNode) node.canvas.parentNode.removeChild(node.canvas);
        delete node.element.dataset.apxRenderer;
        if (state.observer) state.observer.unobserve(node.element);
        // Factor rows are rebuilt on every preset apply, so a window-level
        // listener per node would pile up for the life of the page.
        if (node.releaseWindowEvents) node.releaseWindowEvents();
    }

    function teardown() {
        state.nodes.slice().forEach(unmountNode);
        state.nodes.length = 0;
        state.available = false;
        state.disabled = true;
        state.gl = null;
        document.documentElement.dataset.apxGlassRenderer = 'css';
        if (state.raf) {
            cancelAnimationFrame(state.raf);
            state.raf = 0;
        }
    }

    function bindNodeEvents(node) {
        const element = node.element;
        const target = node.type === 'nav' ? node.host : element;
        const enter = () => { node.hover = true; kick(); };
        const leave = () => { node.hover = false; node.pressed = false; kick(); };
        const down = () => { node.pressed = true; kick(); };
        const up = () => { node.pressed = false; kick(); };
        target.addEventListener('pointerenter', enter);
        target.addEventListener('pointerleave', leave);
        target.addEventListener('pointerdown', down);
        window.addEventListener('pointerup', up);
        node.releaseWindowEvents = () => window.removeEventListener('pointerup', up);
        // A click leaves DOM focus on the control, which would otherwise hold
        // the lens open forever after release. Only keyboard focus counts.
        element.addEventListener('focusin', () => {
            node.focused = state.keyboardNav;
            kick();
        });
        element.addEventListener('focusout', () => { node.focused = false; kick(); });
        element.addEventListener('keydown', () => { node.focused = true; kick(); });
        if (node.type === 'range' || node.type === 'toggle') {
            element.addEventListener('input', () => { node.dirty = true; kick(); });
            element.addEventListener('change', () => { node.dirty = true; kick(); });
        }
        if (node.type === 'nav') {
            target.addEventListener('click', () => { node.dirty = true; kick(); });
        }
    }

    function layoutNode(node) {
        const rect = node.element.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) {
            node.width = 0;
            node.height = 0;
            return;
        }
        const hostRect = node.host.getBoundingClientRect();
        node.width = rect.width;
        node.height = rect.height;
        node.dpr = state.dpr;
        node.sizeScale = clamp(rect.height / REFERENCE_HEIGHT_PX, 0.26, 1.15);

        const cssWidth = rect.width + node.padX * 2;
        const cssHeight = rect.height + node.padY * 2;
        node.canvasWidth = cssWidth;
        node.canvasHeight = cssHeight;
        node.canvasRect = {
            left: rect.left - node.padX,
            top: rect.top - node.padY,
        };

        node.canvas.style.left = `${rect.left - hostRect.left - node.padX}px`;
        node.canvas.style.top = `${rect.top - hostRect.top - node.padY}px`;
        node.canvas.style.width = `${cssWidth}px`;
        node.canvas.style.height = `${cssHeight}px`;

        const bitmapWidth = Math.max(1, Math.round(cssWidth * node.dpr));
        const bitmapHeight = Math.max(1, Math.round(cssHeight * node.dpr));
        if (node.canvas.width !== bitmapWidth) node.canvas.width = bitmapWidth;
        if (node.canvas.height !== bitmapHeight) node.canvas.height = bitmapHeight;
        if (node.scene.width !== bitmapWidth) node.scene.width = bitmapWidth;
        if (node.scene.height !== bitmapHeight) node.scene.height = bitmapHeight;

        const lenses = lensesOf(node.geometry(node));
        node.targets = lenses.map((lens) => lens.x);
        node.targetX = node.targets[0];
        syncLensSizes(node, lenses);
        node.sizes = node.sizeTargets.map((size) => ({ ...size }));
        node.dirty = true;
    }

    function drawScene(node) {
        const ctx = node.sceneContext;
        ctx.setTransform(node.dpr, 0, 0, node.dpr, 0, 0);
        ctx.clearRect(0, 0, node.canvasWidth, node.canvasHeight);
        drawBackdropInto(ctx, node);
        ctx.save();
        ctx.translate(node.padX, node.padY);
        node.paintSurface(ctx, node);
        ctx.restore();
    }

    function renderNode(node) {
        if (!state.available || node.width < 1 || node.height < 1) return;
        const gl = state.gl;
        const dpr = node.dpr;
        const bitmapWidth = node.canvas.width;
        const bitmapHeight = node.canvas.height;

        drawScene(node);
        ensureRenderTarget(bitmapWidth, bitmapHeight);

        gl.useProgram(state.program);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, state.texture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        if (
            state.textureWidth !== node.scene.width
            || state.textureHeight !== node.scene.height
        ) {
            gl.texImage2D(
                gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, node.scene
            );
            state.textureWidth = node.scene.width;
            state.textureHeight = node.scene.height;
        } else {
            gl.texSubImage2D(
                gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, node.scene
            );
        }

        const geometry = node.geometry(node);
        const lenses = lensesOf(geometry);
        const interaction = clamp(node.interaction, 0, 1);
        const scale = node.sizeScale;
        const tuning = node.tuning;

        gl.viewport(0, 0, bitmapWidth, bitmapHeight);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.uniform2f(state.uniforms.resolution, bitmapWidth, bitmapHeight);
        gl.uniform2f(
            state.uniforms.trackCenter,
            (node.padX + geometry.trackX) * dpr,
            (node.padY + geometry.trackY) * dpr
        );
        gl.uniform2f(
            state.uniforms.trackHalfSize,
            geometry.trackHalfWidth * dpr,
            geometry.trackHalfHeight * dpr
        );
        gl.uniform1f(state.uniforms.interaction, interaction);
        gl.uniform1f(state.uniforms.strength, STRENGTH);
        gl.uniform1f(state.uniforms.pixelRatio, dpr);
        gl.uniform1f(state.uniforms.velocity, node.velocityUnit);
        gl.uniform1f(state.uniforms.neutralWeight, node.neutralWeight);
        gl.uniform1f(state.uniforms.shapeMode, node.shape === 'box' ? 1 : 0);
        gl.uniform1f(
            state.uniforms.cornerRadius,
            (geometry.cornerRadius || 0) * dpr
        );
        gl.uniform4f(
            state.uniforms.shapeTuning,
            tuning.edgeIdle * scale,
            tuning.edgeActive * scale,
            tuning.centerShrink,
            tuning.floatingShrink
        );
        gl.uniform4f(
            state.uniforms.innerTuning,
            tuning.innerSpanTrack,
            tuning.innerSpanFloating,
            0,
            0
        );
        gl.uniform4f(
            state.uniforms.blendTuning,
            tuning.innerMixMin,
            tuning.innerMixMax,
            tuning.rimFoldStart,
            tuning.rimFoldMix
        );
        gl.uniform4f(
            state.uniforms.edgeTuning,
            0,
            tuning.edgeCrispStart,
            tuning.edgeCrispEnd,
            tuning.chromaShift * scale
        );
        gl.uniform4f(
            state.uniforms.rimTuning,
            tuning.rimWidth * scale,
            tuning.rimStrength,
            tuning.shoulderWidth * scale,
            tuning.coreWidth * scale
        );
        gl.uniform4f(
            state.uniforms.roundTuning,
            tuning.sideRoundMin,
            tuning.sideRoundMax,
            tuning.chromaMin,
            tuning.chromaMax
        );
        gl.uniform4f(
            state.uniforms.transitionTuning,
            tuning.transitionSpread * scale,
            tuning.transitionDecay,
            tuning.rgbBandWidth * scale,
            tuning.edgeTaper
        );

        // A control can carry more than one lens (the year range has two
        // handles). Only the first pass paints the shared track; the rest are
        // lens-only so they composite on top instead of erasing it.
        lenses.forEach((lens, index) => {
            const grow = lens.grow || { x: 0.16, y: 0.16 };
            // renderLiquidToggle(): the thumb stretches along travel and
            // squashes across it, scaled by how fast it is moving.
            const swell = node.activity;
            const delta = Math.min(12, Math.abs(node.velocityPx));
            const growX = grow.x + delta * (grow.speedX || 0);
            const growY = grow.y - delta * (grow.speedY || 0);
            gl.uniform1f(
                state.uniforms.trackVisibility,
                index === 0 ? node.trackVisibility : 0
            );
            gl.uniform2f(
                state.uniforms.lensCenter,
                (node.padX + (node.positions[index] ?? lens.x)) * dpr,
                (node.padY + (lens.y ?? geometry.lensY)) * dpr
            );
            const eased = node.sizes[index] || lens;
            gl.uniform2f(
                state.uniforms.lensHalfSize,
                Math.max(1, eased.halfWidth * (1 + growX * swell)) * dpr,
                Math.max(1, eased.halfHeight * (1 + growY * swell)) * dpr
            );
            gl.drawArrays(gl.TRIANGLES, 0, 3);
        });

        node.context.setTransform(1, 0, 0, 1, 0, 0);
        node.context.clearRect(0, 0, bitmapWidth, bitmapHeight);
        node.context.drawImage(
            state.glCanvas,
            0,
            state.glCanvas.height - bitmapHeight,
            bitmapWidth,
            bitmapHeight,
            0,
            0,
            bitmapWidth,
            bitmapHeight
        );
    }

    /* ── motion ──────────────────────────────────────────────────────────── */

    // The demo holds the lens open while the thumb is still travelling
    // (lensHeldOpen in updateBarMotion) — that stretch is what makes the
    // slide read as liquid rather than as a moving pill.
    function nodeIsActive(node) {
        const travelling = node.targets.some(
            (target, index) => Math.abs(target - node.positions[index]) > 0.5
        );
        // Hover alone must not hold a switch or slider open. The demo keys off
        // press / drag / travel only, so releasing drops straight back to the
        // flat grey rest state instead of waiting for the pointer to leave.
        return (
            (node.hoverActivates && node.hover)
            || node.pressed
            || node.focused
            || node.dragging
            || travelling
        );
    }

    // A nav tab is a different width from its neighbour, so the lens has to
    // grow/shrink into the new one instead of snapping to it.
    function syncLensSizes(node, lenses) {
        node.sizeTargets = lenses.map((lens) => ({
            halfWidth: lens.halfWidth,
            halfHeight: lens.halfHeight,
        }));
        while (node.sizes.length < lenses.length) {
            node.sizes.push({ ...node.sizeTargets[node.sizes.length] });
        }
        node.sizes.length = lenses.length;
    }

    function easeLensSizes(node, lenses, dt) {
        const rate = 1 - Math.exp(-dt / 48);
        let settled = true;
        node.sizes.forEach((size, index) => {
            const target = node.sizeTargets[index];
            ['halfWidth', 'halfHeight'].forEach((key) => {
                const distance = target[key] - size[key];
                if (Math.abs(distance) >= 0.05) {
                    size[key] += distance * rate;
                    settled = false;
                } else {
                    size[key] = target[key];
                }
            });
        });
        return settled;
    }

    function updateMotion(node, dt) {
        const geometry = node.geometry(node);
        const lenses = lensesOf(geometry);
        node.targets = lenses.map((lens) => lens.x);
        while (node.positions.length < node.targets.length) {
            node.positions.push(node.targets[node.positions.length]);
        }
        node.positions.length = node.targets.length;
        node.targetX = node.targets[0];
        syncLensSizes(node, lenses);
        const desired = nodeIsActive(node) ? 1 : node.idleInteraction;

        if (state.reduceMotion) {
            const moved = node.targets.some(
                (target, index) => Math.abs(target - node.positions[index]) > 0.05
            ) || Math.abs(desired - node.interaction) > 0.002;
            node.interaction = desired;
            node.activity = desired;
            node.positions = node.targets.slice();
            node.x = node.positions[0];
            node.sizes = node.sizeTargets.map((size) => ({ ...size }));
            node.velocityUnit = 0;
            node.velocityPx = 0;
            return moved;
        }

        const frameScale = clamp(dt / 16.667, 0.5, 2);
        const previousX = node.positions[0];
        if (node.springMotion) {
            // updateLiquidToggle() in the demo: a spring that overshoots to
            // 1.16 on engage and decays on release. The overshoot is the pop.
            if (desired > node.idleInteraction) {
                node.activityVelocity += (1 - node.activity) * 0.25 * frameScale;
                node.activityVelocity *= Math.pow(0.68, frameScale);
                node.activity = Math.min(
                    1.16,
                    node.activity + node.activityVelocity * frameScale
                );
            } else {
                node.activityVelocity = 0;
                node.activity += (node.idleInteraction - node.activity)
                    * (1 - Math.exp(-dt / 62));
            }
            node.interaction = Math.min(1, node.activity);
        } else {
            const rate = desired > node.interaction ? 0.38 : 0.22;
            node.interaction += (desired - node.interaction) * rate * frameScale;
            node.activity = node.interaction;
        }

        let settled = easeLensSizes(node, lenses, dt);
        node.targets.forEach((target, index) => {
            const distance = target - node.positions[index];
            if (Math.abs(distance) >= 0.05) {
                node.positions[index] += distance * (1 - Math.exp(-dt / 48));
                settled = false;
            } else {
                node.positions[index] = target;
            }
        });
        node.x = node.positions[0];
        node.velocityPx = node.x - previousX;

        const travel = Math.max(node.width, 1);
        node.velocityUnit += (
            clamp((node.x - previousX) / (travel * 0.08), -1, 1) - node.velocityUnit
        ) * 0.45;
        if (Math.abs(node.velocityUnit) < 0.002) node.velocityUnit = 0;

        const targetActivity = node.springMotion && desired > node.idleInteraction
            ? 1
            : desired;
        return (
            Math.abs(targetActivity - node.activity) > 0.002
            || !settled
            || node.velocityUnit !== 0
        );
    }

    function stepFrame(dt) {
        if (state.backdropDirty) paintBackdrop();

        let running = false;
        state.frames += 1;
        state.nodes.forEach((node) => {
            if (!node.mounted || !node.visible || node.width < 1) return;
            const animating = updateMotion(node, dt);
            if (node.syncDom) node.syncDom(node);
            if (animating || node.dirty) {
                renderNode(node);
                node.dirty = false;
                state.renders += 1;
            }
            node.animating = animating;
            if (animating) running = true;
        });
        return running;
    }

    function tick(now) {
        const dt = clamp(now - state.lastTime, 1, 64);
        state.lastTime = now;
        const running = stepFrame(dt);
        state.raf = running ? requestAnimationFrame(tick) : 0;
    }

    function kick() {
        if (!state.available || state.raf) return;
        state.lastTime = performance.now();
        state.raf = requestAnimationFrame(tick);
    }

    function invalidate(relayout = false) {
        if (!state.available) return;
        readTheme();
        state.applying = true;
        state.backdropDirty = true;
        state.nodes.forEach((node) => {
            if (relayout) layoutNode(node);
            node.dirty = true;
        });
        state.applying = false;
        kick();
    }

    /* ── surface painters ────────────────────────────────────────────────── */

    // drawBarScene()'s quietSelection: at rest the demo shows a plain solid
    // thumb and no glass at all — the lens only takes over as the control is
    // engaged, and the solid thumb fades out underneath it.
    function quietWeight(node) {
        return Math.max(0, 1 - Math.min(1, node.activity * 1.35));
    }

    function solidThumbColor(progress, alpha) {
        const off = state.scene.thumbOff;
        const on = state.scene.thumbOn;
        const channels = off.map(
            (value, index) => Math.round(value + (on[index] - value) * progress)
        );
        return `rgba(${channels.join(', ')}, ${alpha.toFixed(3)})`;
    }

    const LENS_AA_COVER_PX = 1.5;

    function paintQuietThumb(ctx, node, x, y, halfWidth, halfHeight, color, rim, radius) {
        const quiet = quietWeight(node);
        if (quiet <= 0.001) return;
        // cover the lens mask's antialiased edge, otherwise a feathered ring of
        // bare backdrop shows around the thumb
        const spreadX = halfWidth + LENS_AA_COVER_PX;
        const spreadY = halfHeight + LENS_AA_COVER_PX;
        const path = () => roundedRectPath(
            ctx,
            x - spreadX,
            y - spreadY,
            spreadX * 2,
            spreadY * 2,
            radius == null
                ? Math.min(spreadX, spreadY)
                : radius + LENS_AA_COVER_PX
        );
        path();
        ctx.fillStyle = color(quiet);
        ctx.fill();
        if (!rim) return;
        // a thin rim keeps a white thumb legible on a green ON track
        path();
        ctx.strokeStyle = rim;
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // No bar frame and no canvas-drawn labels: the nav is bare text with a
    // selection lens sliding behind it.
    function paintNavSurface(ctx, node) {
        const geometry = node.geometry(node);
        paintQuietThumb(
            ctx,
            node,
            node.x,
            geometry.lensY,
            geometry.halfWidth,
            geometry.halfHeight,
            (quiet) => `rgba(${state.scene.navQuiet.join(', ')}, ${(quiet * 0.88).toFixed(3)})`
        );
    }

    // The thumb swells past the pill while engaged. Only the lens is masked
    // out there, so without a bleed of track colour beyond the pill it
    // refracts the page behind it — a black slab in dark mode, white in light.
    const TRACK_BLEED_PX = 10;

    function paintTogglePillSurface(ctx, node) {
        const progress = node.progress(node);
        const radius = node.height / 2;
        const fillPill = (inset) => {
            roundedRectPath(
                ctx,
                inset,
                inset,
                node.width - inset * 2,
                node.height - inset * 2,
                radius - inset
            );
            ctx.fillStyle = state.scene.pill;
            ctx.fill();
            if (progress > 0.001) {
                ctx.save();
                ctx.globalAlpha = progress;
                ctx.fillStyle = state.scene.pillOn;
                ctx.fill();
                ctx.restore();
            }
        };
        // painted first and hidden by the track mask until the lens grows onto it
        fillPill(-TRACK_BLEED_PX);
        fillPill(0);
        roundedRectPath(ctx, 0, 0, node.width, node.height, radius);
        ctx.strokeStyle = state.scene.pillStroke;
        ctx.lineWidth = 1;
        ctx.stroke();

        // The DOM thumb fades out as the lens fades in, so the scene has to
        // take the thumb over — otherwise holding the switch refracts a flat
        // track and the ball simply disappears.
        const glass = Math.min(1, node.activity);
        if (glass > 0.001) {
            const { thumbWidth, thumbHeight } = toggleTrack(node);
            const halfW = thumbWidth / 2;
            const halfH = thumbHeight / 2;
            roundedRectPath(
                ctx,
                node.x - halfW,
                node.height / 2 - halfH,
                thumbWidth,
                thumbHeight,
                halfH
            );
            ctx.fillStyle = solidThumbColor(progress, glass * 0.98);
            ctx.fill();
        }

    }

    // Slim track, glass thumb — a fat capsule reads as a progress bar in a
    // sidebar this dense.
    const TRACK_THICKNESS_PX = 9;

    function trackBox(node) {
        const top = (node.height - TRACK_THICKNESS_PX) / 2;
        return { top, height: TRACK_THICKNESS_PX, radius: TRACK_THICKNESS_PX / 2 };
    }

    function paintTrack(ctx, node, fillFrom, fillTo, accent) {
        const track = trackBox(node);
        roundedRectPath(ctx, 0, track.top, node.width, track.height, track.radius);
        ctx.fillStyle = state.scene.track;
        ctx.fill();

        ctx.save();
        roundedRectPath(ctx, 0, track.top, node.width, track.height, track.radius);
        ctx.clip();
        ctx.fillStyle = accent;
        ctx.fillRect(
            clamp(fillFrom, 0, node.width),
            track.top,
            clamp(fillTo - fillFrom, 0, node.width),
            track.height
        );
        ctx.restore();

        roundedRectPath(ctx, 0, track.top, node.width, track.height, track.radius);
        ctx.strokeStyle = state.scene.trackStroke;
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // demo range thumb proportions (65x42), scaled to the APX track
    const DUAL_GRIP_HALF_W = 13;
    const DUAL_GRIP_HALF_H = 8;

    function paintRangeThumbs(ctx, node, halfWidth, halfHeight, radius) {
        node.positions.forEach((x) => {
            paintQuietThumb(
                ctx,
                node,
                x,
                node.height / 2,
                halfWidth,
                halfHeight,
                (quiet) => `rgba(${state.scene.thumbOff.join(', ')}, ${(quiet * 0.96).toFixed(3)})`,
                state.scene.thumbRim,
                radius
            );
        });
    }

    function paintRangeSurface(ctx, node) {
        paintTrack(ctx, node, 0, node.x, 'rgba(0, 229, 160, 0.92)');
        paintRangeThumbs(ctx, node, 13, 9);
    }

    function paintDualRangeSurface(ctx, node) {
        paintTrack(
            ctx,
            node,
            node.positions[0] ?? 0,
            node.positions[1] ?? node.width,
            'rgba(255, 167, 38, 0.92)'
        );
        paintRangeThumbs(ctx, node, DUAL_GRIP_HALF_W, DUAL_GRIP_HALF_H);
    }

    function accentOf(element) {
        if (element.classList.contains('apx-glass-action--green')) return state.theme.green;
        if (element.classList.contains('apx-glass-action--amber')) return state.theme.amber;
        if (element.classList.contains('on')) return state.theme.green;
        if (element.classList.contains('connected')) return state.theme.green;
        if (element.classList.contains('funded')) return state.theme.amber;
        return null;
    }

    function paintSurfaceFill(ctx, node) {
        const geometry = node.geometry(node);
        const radius = node.shape === 'box'
            ? geometry.cornerRadius
            : node.height / 2;
        roundedRectPath(ctx, 0, 0, node.width, node.height, radius);
        ctx.fillStyle = state.scene.surface;
        ctx.fill();

        // Accent covers the whole button — no radial falloff leaving the right
        // half uncoloured.
        const accent = accentOf(node.element);
        if (accent) {
            ctx.save();
            roundedRectPath(ctx, 0, 0, node.width, node.height, radius);
            ctx.clip();
            ctx.globalAlpha = 0.30;
            ctx.fillStyle = accent;
            ctx.fillRect(0, 0, node.width, node.height);
            ctx.restore();
        }
    }

    /* ── adapters ────────────────────────────────────────────────────────── */

    // The canvas has to live in a positioned box that is NOT the control
    // itself: several controls have their text content rewritten at runtime
    // (toggleField writes button.textContent), which would delete a child
    // canvas. `display` is passed per adapter rather than sniffed — a block
    // mount keeps width:100% controls filling their container, an inline one
    // keeps a checkbox sitting on the same line as its label text.
    function ensureMount(element, display) {
        const parent = element.parentNode;
        if (parent && parent.classList && parent.classList.contains('apx-glass-mount')) {
            return parent;
        }
        const style = getComputedStyle(element);
        const mount = document.createElement('span');
        mount.className = 'apx-glass-mount';
        if (display) mount.style.display = display;
        const parentStyle = parent ? getComputedStyle(parent) : null;
        if (parentStyle && parentStyle.display.includes('flex')) {
            mount.style.flex = `${style.flexGrow} ${style.flexShrink} ${style.flexBasis}`;
            mount.style.alignSelf = style.alignSelf;
        }
        parent.insertBefore(mount, element);
        mount.appendChild(element);
        return mount;
    }

    function mountNav(root) {
        if (root.dataset.apxRenderer === 'webgl') return;
        const selector = root.classList.contains('bottom-tabs')
            ? '.bottom-tab'
            : '.tab';
        if (!root.querySelector(selector)) return;
        // .header-tabs is absolutely positioned and centred — only promote a
        // static nav, never overwrite an existing positioning scheme.
        if (getComputedStyle(root).position === 'static') {
            root.style.position = 'relative';
        }

        const segments = () => {
            const rootRect = root.getBoundingClientRect();
            return [...root.querySelectorAll(selector)].map((tab) => {
                const rect = tab.getBoundingClientRect();
                const style = getComputedStyle(tab);
                const label = style.textTransform === 'uppercase'
                    ? tab.textContent.trim().toUpperCase()
                    : tab.textContent.trim();
                return {
                    element: tab,
                    active: tab.classList.contains('active'),
                    left: rect.left - rootRect.left,
                    top: rect.top - rootRect.top,
                    width: rect.width,
                    height: rect.height,
                    label,
                    font: `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`,
                    letterSpacing: style.letterSpacing === 'normal'
                        ? '0px'
                        : style.letterSpacing,
                };
            });
        };

        const node = createNode({
            type: 'nav',
            element: root,
            host: root,
            padX: 44,
            padY: 34,
            shape: 'capsule',
            // no bar behind the tabs — only the selection lens is painted
            trackVisibility: 0,
            neutralWeight: 1,
            idleInteraction: 0,
            hoverActivates: false,
            segments,
            paintSurface: paintNavSurface,
            geometry: (node) => {
                const list = node.segments();
                const held = node.dragging && node.dragSegment
                    ? node.dragSegment
                    : null;
                const active = held
                    || list.find((segment) => segment.active)
                    || list[0]
                    || { left: 0, top: 0, width: node.width, height: node.height };
                // renderBar()'s switch geometry: the lens swells past the tab
                // in both axes while it is engaged.
                const scale = node.sizeScale;
                const idleHalfHeight = Math.max(9, node.height / 2 - 6 * scale);
                const dragHalfHeight = node.height / 2 + 15 * scale;
                const idleHalfWidth = Math.max(
                    active.width * 0.48,
                    idleHalfHeight + 1
                );
                const dragHalfWidth = Math.max(
                    active.width * 0.60,
                    dragHalfHeight + 1
                );
                const engaged = clamp(node.interaction, 0, 1);
                return {
                    trackX: node.width / 2,
                    trackY: node.height / 2,
                    trackHalfWidth: node.width / 2,
                    trackHalfHeight: node.height / 2,
                    lensX: node.dragging && node.dragX != null
                        ? clamp(node.dragX, active.width / 2, node.width - active.width / 2)
                        : active.left + active.width / 2,
                    lensY: active.top + active.height / 2,
                    halfWidth: idleHalfWidth + (dragHalfWidth - idleHalfWidth) * engaged,
                    halfHeight: idleHalfHeight + (dragHalfHeight - idleHalfHeight) * engaged,
                    grow: { x: 0, y: 0, speedX: 0.006, speedY: 0.006 },
                };
            },
        });

        bindNavDrag(node, root, selector);
    }

    // Drag the nav thumb the way the demo's segmented switch does: the lens
    // follows the pointer, and the segment under it commits on release.
    function bindNavDrag(node, root, selector) {
        const localX = (event) => event.clientX - root.getBoundingClientRect().left;
        const segmentAt = (x) => node.segments().find(
            (segment) => x >= segment.left && x <= segment.left + segment.width
        );

        const move = (event) => {
            if (!node.dragging) return;
            event.preventDefault();
            node.dragX = clamp(localX(event), 0, node.width);
            node.dragSegment = segmentAt(node.dragX) || node.dragSegment;
            kick();
        };

        const end = (event) => {
            if (!node.dragging) return;
            node.dragging = false;
            const target = segmentAt(
                node.dragX == null ? localX(event) : node.dragX
            );
            node.dragX = null;
            node.dragSegment = null;
            if (target && !target.active) target.element.click();
            node.dirty = true;
            kick();
        };

        root.addEventListener('pointerdown', (event) => {
            if (event.button !== 0) return;
            node.dragging = true;
            node.dragX = localX(event);
            node.dragSegment = segmentAt(node.dragX);
            kick();
        });
        window.addEventListener('pointermove', move);
        window.addEventListener('pointerup', end);
        window.addEventListener('pointercancel', end);

        const release = node.releaseWindowEvents;
        node.releaseWindowEvents = () => {
            if (release) release();
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', end);
            window.removeEventListener('pointercancel', end);
        };
    }

    // Bench proportions: a wide capsule thumb, not a circle.
    const TOGGLE_THUMB_ASPECT = 1.45;

    function toggleTrack(node) {
        const inset = 4;
        const height = Math.max(6, node.height - inset * 2);
        const width = Math.min(
            height * TOGGLE_THUMB_ASPECT,
            Math.max(height, node.width - inset * 2 - 8)
        );
        return {
            inset,
            thumbHeight: height,
            thumbWidth: width,
            travel: Math.max(1, node.width - inset * 2 - width),
        };
    }

    function mountToggle(input) {
        if (input.dataset.apxRenderer === 'webgl') return;
        const mount = ensureMount(input, 'inline-flex');
        mount.classList.add('apx-switch-shell');
        mount.style.flex = '0 0 auto';
        mount.style.verticalAlign = 'middle';
        // The demo layers track (z0) → glass canvas (z1) → solid thumb (z2),
        // cross-fading canvas and thumb with --toggle-glass. Track and thumb
        // stay in the DOM, so no theme rule can ever paint over the surface.
        const track = document.createElement('span');
        track.className = 'apx-switch-track';
        track.setAttribute('aria-hidden', 'true');
        const thumb = document.createElement('span');
        thumb.className = 'apx-switch-thumb';
        thumb.setAttribute('aria-hidden', 'true');
        mount.insertBefore(track, mount.firstChild);
        mount.appendChild(thumb);

        const node = createNode({
            type: 'toggle',
            element: input,
            host: mount,
            padX: 22,
            padY: 22,
            shape: 'capsule',
            // canvas paints the lens only; the track is a DOM element
            trackVisibility: 0,
            neutralWeight: 0,
            idleInteraction: 0,
            springMotion: true,
            hoverActivates: false,
            tuning: { centerShrink: 0.32 },
            progress: (node) => (
                node.dragging && node.dragProgress != null
                    ? node.dragProgress
                    : (node.element.checked ? 1 : 0)
            ),
            paintSurface: paintTogglePillSurface,
            syncDom: syncToggleDom,
            geometry: (node) => {
                const { inset, thumbWidth, thumbHeight, travel } = toggleTrack(node);
                return {
                    trackX: node.width / 2,
                    trackY: node.height / 2,
                    trackHalfWidth: node.width / 2,
                    trackHalfHeight: node.height / 2,
                    lensX: inset + thumbWidth / 2 + node.progress(node) * travel,
                    lensY: node.height / 2,
                    halfWidth: thumbWidth / 2,
                    halfHeight: thumbHeight / 2,
                    // renderLiquidToggle() stretch/squash constants
                    grow: { x: 0.65, y: 0.65, speedX: 0.025, speedY: 0.025 },
                };
            },
        });
        bindToggleDrag(node, input);
    }

    // renderLiquidToggle(): drive the DOM track/thumb from the same state the
    // lens uses, and cross-fade the two.
    function syncToggleDom(node) {
        const { inset, thumbWidth, thumbHeight } = toggleTrack(node);
        const progress = node.progress(node);
        const glass = Math.min(1, node.activity);
        const delta = Math.min(12, Math.abs(node.velocityPx));
        const width = thumbWidth * (1 + (0.65 + delta * 0.025) * node.activity);
        const height = thumbHeight * (1 + (0.65 - delta * 0.025) * node.activity);
        const anchor = inset + thumbWidth / 2;
        const style = node.host.style;
        style.setProperty('--apx-switch-progress', progress.toFixed(4));
        style.setProperty('--apx-switch-anchor', `${anchor.toFixed(2)}px`);
        style.setProperty('--apx-switch-travel', `${(node.x - anchor).toFixed(2)}px`);
        style.setProperty('--apx-switch-w', `${width.toFixed(2)}px`);
        style.setProperty(
            '--apx-switch-h',
            `${Math.max(thumbHeight, height).toFixed(2)}px`
        );
        style.setProperty('--apx-switch-glass', glass.toFixed(4));
        style.setProperty('--apx-switch-solid', solidThumbColor(progress, 1));
        style.setProperty('--apx-switch-off', state.scene.pill);
        style.setProperty('--apx-switch-on', state.scene.pillOn);
    }

    // Slide the thumb left/right like the demo toggle instead of only
    // accepting a click. A drag commits on release and swallows the click the
    // browser fires afterwards, so the value does not flip twice.
    function bindToggleDrag(node, input) {
        const progressAt = (event) => {
            const rect = input.getBoundingClientRect();
            const { inset, thumb, travel } = toggleTrack(node);
            return clamp(
                (event.clientX - rect.left - inset - thumb / 2) / travel,
                0,
                1
            );
        };

        const move = (event) => {
            if (!node.dragging) return;
            if (Math.abs(event.clientX - node.dragOriginX) > 3) {
                node.dragMoved = true;
            }
            node.dragProgress = progressAt(event);
            kick();
        };

        const end = () => {
            if (!node.dragging) return;
            node.dragging = false;
            const settled = node.dragProgress;
            node.dragProgress = null;
            if (node.dragMoved) {
                const next = settled > 0.5;
                input.addEventListener('click', swallowClick, { capture: true, once: true });
                if (next !== input.checked) {
                    input.checked = next;
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
            node.dragMoved = false;
            node.dirty = true;
            kick();
        };

        const swallowClick = (event) => {
            event.preventDefault();
            event.stopPropagation();
        };

        input.addEventListener('pointerdown', (event) => {
            if (event.button !== 0 || input.disabled) return;
            node.dragging = true;
            node.dragMoved = false;
            node.dragOriginX = event.clientX;
            node.dragProgress = input.checked ? 1 : 0;
            kick();
        });
        window.addEventListener('pointermove', move);
        window.addEventListener('pointerup', end);
        window.addEventListener('pointercancel', end);

        const release = node.releaseWindowEvents;
        node.releaseWindowEvents = () => {
            if (release) release();
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', end);
            window.removeEventListener('pointercancel', end);
        };
    }

    function inputRatio(input) {
        const min = Number(input.min || 0);
        const max = Number(input.max || 100);
        const value = Number(input.value || min);
        return max > min ? clamp((value - min) / (max - min), 0, 1) : 0;
    }

    function rangeTrackGeometry(node) {
        const track = trackBox(node);
        return {
            trackX: node.width / 2,
            trackY: node.height / 2,
            trackHalfWidth: node.width / 2,
            trackHalfHeight: track.height / 2,
            lensY: node.height / 2,
        };
    }

    function mountRange(input) {
        if (input.dataset.apxRenderer === 'webgl') return;
        const mount = ensureMount(input);
        createNode({
            type: 'range',
            element: input,
            host: mount,
            padX: 30,
            padY: 24,
            shape: 'capsule',
            trackVisibility: 1,
            neutralWeight: 0,
            idleInteraction: 0,
            hoverActivates: false,
            paintSurface: paintRangeSurface,
            geometry: (node) => {
                const halfWidth = 13;
                const halfHeight = 9;
                return {
                    ...rangeTrackGeometry(node),
                    lensX: clamp(
                        inputRatio(node.element) * node.width,
                        halfWidth,
                        Math.max(halfWidth, node.width - halfWidth)
                    ),
                    halfWidth,
                    halfHeight,
                    grow: { x: 0.34, y: 0.26, speedX: 0.018, speedY: 0.018 },
                };
            },
        });
    }

    // The year range is two overlaid <input type=range> handles sharing one
    // track, so it mounts as a single node with two lenses.
    function mountDualRange(root) {
        if (root.dataset.apxRenderer === 'webgl') return;
        const inputs = [...root.querySelectorAll('input[type=range]')];
        if (inputs.length < 2) return;
        if (getComputedStyle(root).position === 'static') {
            root.style.position = 'relative';
        }
        createNode({
            type: 'dual-range',
            element: root,
            host: root,
            padX: 30,
            padY: 24,
            shape: 'capsule',
            trackVisibility: 1,
            neutralWeight: 0,
            idleInteraction: 0,
            hoverActivates: false,
            paintSurface: paintDualRangeSurface,
            geometry: (node) => {
                const halfWidth = DUAL_GRIP_HALF_W;
                const halfHeight = DUAL_GRIP_HALF_H;
                const place = (input) => clamp(
                    inputRatio(input) * node.width,
                    halfWidth,
                    Math.max(halfWidth, node.width - halfWidth)
                );
                return {
                    ...rangeTrackGeometry(node),
                    lenses: inputs.map((input) => ({
                        x: place(input),
                        halfWidth,
                        halfHeight,
                        grow: { x: 0.34, y: 0.26, speedX: 0.018, speedY: 0.018 },
                    })),
                };
            },
        });
        inputs.forEach((input) => {
            input.addEventListener('input', () => { kick(); });
            input.addEventListener('change', () => { kick(); });
        });
    }

    function mountSurface(element, shape, options = {}) {
        if (element.dataset.apxRenderer === 'webgl') return;
        const mount = ensureMount(element);
        createNode({
            type: 'surface',
            hoverActivates: options.hoverActivates !== false,
            padX: options.pad ?? 14,
            padY: options.pad ?? 14,
            element,
            host: mount,
            shape,
            trackVisibility: 0,
            neutralWeight: 1,
            idleInteraction: 0.34,
            // A button must not change shape on hover: zero lens growth, and
            // zero content shrink so the fill does not pull away from the edge.
            tuning: { centerShrink: 0, floatingShrink: 0 },
            paintSurface: paintSurfaceFill,
            geometry: (node) => {
                const cornerRadius = node.shape === 'box'
                    ? (options.cornerRadius ?? Math.min(18, node.height * 0.34))
                    : 0;
                return {
                    trackX: node.width / 2,
                    trackY: node.height / 2,
                    trackHalfWidth: node.width / 2,
                    trackHalfHeight: node.height / 2,
                    lensX: node.width / 2,
                    lensY: node.height / 2,
                    halfWidth: node.width / 2,
                    halfHeight: node.height / 2,
                    cornerRadius,
                    grow: { x: 0, y: 0 },
                };
            },
        });
    }

    /* ── mount sweep ─────────────────────────────────────────────────────── */

    const TARGETS = [
        { selector: '.apx-glass-nav', mount: (el) => mountNav(el) },
        {
            // factor rows and the header light/dark switch
            selector: '.apx-glass-switch input[type=checkbox]',
            mount: (el) => mountToggle(el),
        },
        {
            selector: 'input[type=range].apx-glass-range',
            mount: (el) => mountRange(el),
        },
        { selector: '.dual-range', mount: (el) => mountDualRange(el) },
        {
            // real glass containers, not a backdrop-filter imitation
            selector: '.panel.apx-glass-container',
            mount: (el) => mountSurface(el, 'box', {
                cornerRadius: 22,
                pad: 10,
                hoverActivates: false,
            }),
        },
        {
            selector: '.btn.apx-glass-action',
            mount: (el) => mountSurface(el, 'box'),
        },
        {
            selector: '.toggle-field.apx-glass-toggle',
            mount: (el) => mountSurface(el, 'capsule'),
        },
        {
            selector: '.conn-trigger.apx-glass-status, .account-badge.apx-glass-badge',
            mount: (el) => mountSurface(el, 'capsule'),
        },
    ];

    function refresh() {
        if (!initRenderer()) return;
        state.applying = true;
        state.nodes.slice().forEach((node) => {
            if (!node.element.isConnected) {
                unmountNode(node);
                state.nodes.splice(state.nodes.indexOf(node), 1);
            }
        });
        TARGETS.forEach(({ selector, mount }) => {
            document.querySelectorAll(selector).forEach((element) => {
                if (element.dataset.apxRenderer === 'webgl') return;
                mount(element);
                if (state.observer) state.observer.observe(element);
            });
        });
        state.applying = false;
        invalidate(true);
    }

    // Chrome that ticks on a timer must not drag the whole renderer with it:
    // the UTC clock rewrites its text every second, and a refresh per second
    // would repaint every glass node forever.
    const MUTATION_IGNORE = '#clock, .log, .apx-glass-canvas, .apx-glass-mount';

    function isIgnorableMutation(record) {
        const target = record.target.nodeType === Node.ELEMENT_NODE
            ? record.target
            : record.target.parentElement;
        return !target || Boolean(target.closest(MUTATION_IGNORE));
    }

    // Sidebar/header content is rebuilt at runtime (factor rows, metric cards,
    // workspace switches). Re-sweep on a debounce instead of asking every call
    // site to remember to notify the renderer.
    function scheduleRefresh(records) {
        if (state.applying) return;
        if (records && records.every(isIgnorableMutation)) return;
        window.clearTimeout(state.refreshTimer);
        state.refreshTimer = window.setTimeout(refresh, 80);
    }

    /* ── bootstrap ───────────────────────────────────────────────────────── */

    function init() {
        state.dpr = getDpr();
        state.reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        readTheme();
        if (!initRenderer()) return;

        if ('IntersectionObserver' in window) {
            state.observer = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    const node = state.nodes.find((item) => item.element === entry.target);
                    if (!node) return;
                    node.visible = entry.isIntersecting;
                    if (node.visible) {
                        node.dirty = true;
                        kick();
                    }
                });
            }, { rootMargin: '96px' });
        }

        refresh();

        window.addEventListener('resize', () => {
            state.dpr = getDpr();
            invalidate(true);
        });
        window.addEventListener('keydown', () => { state.keyboardNav = true; }, true);
        window.addEventListener('pointerdown', () => { state.keyboardNav = false; }, true);
        document.addEventListener('scroll', onScroll, true);
        window.addEventListener('focus', () => invalidate(true));
        // rAF is paused while the tab is hidden; repaint once it comes back so
        // no control is left showing a stale (or unrendered) surface.
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) invalidate(true);
        });
        if (document.fonts && document.fonts.ready) {
            document.fonts.ready.then(() => invalidate(true));
        }

        if ('ResizeObserver' in window) {
            const resizeObserver = new ResizeObserver(() => invalidate(true));
            resizeObserver.observe(document.body);
            ['.sidebar', '.bottom-panel', '.header'].forEach((selector) => {
                const region = document.querySelector(selector);
                if (region) resizeObserver.observe(region);
            });
        }

        const mutationObserver = new MutationObserver(scheduleRefresh);
        ['.header', '.sidebar', '.bottom-tabs'].forEach((selector) => {
            const region = document.querySelector(selector);
            if (!region) return;
            mutationObserver.observe(region, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['class'],
            });
        });
    }

    function onScroll() {
        if (!state.available) return;
        window.clearTimeout(state.scrollTimer);
        state.scrollTimer = window.setTimeout(() => invalidate(true), 90);
    }

    function describeNode(node) {
        return {
            type: node.type,
            id: node.element.id || String(node.element.className || ''),
            visible: node.visible,
            width: Number(node.width.toFixed(1)),
            interaction: Number(node.interaction.toFixed(4)),
            activity: Number(node.activity.toFixed(4)),
            activityVelocity: Number(node.activityVelocity.toFixed(5)),
            velocityPx: Number(node.velocityPx.toFixed(3)),
            x: Number(node.x.toFixed(2)),
            targetX: Number(node.targetX.toFixed(2)),
        };
    }

    window.ApxGlass = {
        refresh,
        invalidate,
        // QA hook. requestAnimationFrame is paused while the document is
        // hidden, so headless checks need a deterministic way to advance the
        // renderer (the demo exposes __liquidGlassDiagnostics for the same
        // reason). Not used by the app itself.
        step(frames = 1, dt = 16.667) {
            let running = false;
            for (let index = 0; index < frames; index += 1) {
                running = stepFrame(dt);
            }
            return running;
        },
        get diagnostics() {
            return {
                renderer: state.available ? 'webgl' : 'css',
                nodes: state.nodes.length,
                contexts: state.available ? 1 : 0,
                contextFailures: state.contextFailures,
                dpr: state.dpr,
                frames: state.frames,
                renders: state.renders,
                idle: state.raf === 0,
                animating: state.nodes
                    .filter((node) => node.animating)
                    .map(describeNode),
                all: state.nodes.map(describeNode),
            };
        },
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();
