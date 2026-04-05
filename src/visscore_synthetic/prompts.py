"""Text prompts for Tufte-aligned vs chartjunk diffusion generation."""

TUFE_PROMPTS: list[str] = [
    "Clean minimalist line chart, Edward Tufte style, high data-ink ratio, faint or no gridlines, "
    "no chartjunk, flat 2D only, small sans-serif axis labels, muted neutral colors, white background, "
    "professional editorial statistics graphic",
    "Sparse scatter plot, Tufte-inspired design, maximum data emphasis, thin axes, no decorative borders, "
    "no 3D, restrained grayscale and one accent color, plenty of whitespace, scientific publication quality",
    "Simple bar chart, minimalist data visualization, high data-ink, subtle tick marks only, "
    "no icons or clipart, flat colors, clear typographic hierarchy, light gray grid optional, white paper",
    "Small multiples panel of tiny line charts, consistent scales, Tufte small-multiples style, "
    "dense data sparse ink, no drop shadows, no gradients, monochrome with minimal ink",
    "Horizontal bar ranking chart, clean Tufte aesthetic, direct labeling where possible, "
    "no heavy frames, 2D flat, neutral palette, institutional report style",
    "Minimal area chart with thin stroke, white background, understated axes, no legends box if avoidable, "
    "Edward Tufte principles, no embellishment, high clarity",
    "Dot plot / strip plot style visualization, simple geometry, thin rules, small text, "
    "no neon, no 3D pie, serious analytic look",
    "Sparkline row beneath small table, Tufte sparkline style, tiny high-resolution trend lines, "
    "gray baseline only, no chart decoration",
    "Dual-axis avoided; single clear line chart with modest grid, Tufte data-ink focus, "
    "sans-serif captions, calm colors, print-ready infographic",
    "Column chart with narrow bars, white space between groups, minimal axis lines, "
    "no gradients or glow, flat 2D, textbook Tufte compliance",
    "Time series with light vertical reference lines only, sparse ink, readable micro-labels, "
    "no pictograms, neutral tones, executive briefing chart",
    "Box plot summary, minimal ink, thin whiskers, small median marks, gray axes on white, "
    "no 3D, academic Tufte-style figure",
]

NON_TUFE_PROMPTS: list[str] = [
    "Extremely busy dashboard chart, heavy chartjunk, thick glowing borders, neon cyan and magenta, "
    "3D extruded bar chart, lens flare style highlights, decorative icons, dark gamer UI chrome",
    "Cluttered pie chart with exploded slices, drop shadows, bevel and emboss, saturated rainbow colors, "
    "heavy gradient fills, clipart dollar signs, 3D perspective, misleading emphasis",
    "Overdesigned infographic bar chart, skeuomorphic metal textures, unnecessary 3D rotation, "
    "busy background pattern, huge decorative title banner, chartjunk everywhere",
    "Flashy line chart with thick gradient area fill, glowing neon grid, starburst icons, "
    "heavy drop shadow under plot, busy textured backdrop, cyberpunk dashboard widget",
    "3D column chart with fake depth and perspective distortion, rainbow color cycle per bar, "
    "thick white outlines, cartoon mascot in corner, cluttered legend boxes",
    "Messy combo chart, too many y-axis colors, decorative arrows and callouts, "
    "heavy frame with rounded corners, glossy glass effect, unnecessary pictograms",
    "Pie chart with 12 rainbow slices, 3D tilt, exploded segments, shadows, "
    "sparkle effects, busy legend with icons, poster-style clutter",
    "Dark mode chart with loud gradients, animated-style glow on bars, hex grid background noise, "
    "futuristic HUD ornaments, low data-ink flashy visualization",
    "Stock-photo style infographic chart, watermarks, huge 3D numbers floating, "
    "clipart people pointing, excessive arrows, neon outlines",
    "Dashboard tile with fake leather texture, embossed buttons, 3D donut charts, "
    "busy KPI badges, chartjunk borders and stickers",
    "Overlapping translucent layers, misaligned 3D bars, rainbow laser gradients, "
    "decorative corner flourishes, unreadable tiny 3D text",
    "Skeuomorphic thermometer chart, glossy plastic tubes, unnecessary 3D, "
    "lens reflections, busy wallpaper background, comic sans style clutter",
]
