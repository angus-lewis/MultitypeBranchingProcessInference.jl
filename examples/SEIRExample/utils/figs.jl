const FIGURE_FILE_EXT = "pdf"
default(; fontfamily="Bookman", size=(400,300), legendfontsize=10, guidefontsize=12, tickfontsize=10, titlefontsize=14, grid=false)

const _pastel_blue = RGB(0.4, 0.5, 0.8)
const _pastel_green = RGB(0.4, 0.8, 0.4)
const _pastel_red = RGB(0.8, 0.4, 0.4)
const _gray = RGB(0.5, 0.5, 0.5)

const COLOUR_MAP = Dict(
    1 => _pastel_blue,
    2 => _pastel_green,
    3 => _pastel_red,
    4 => :grey,
    5 => :purple,
    6 => :magenta,
)
cmap(i) = (i in keys(COLOUR_MAP)) ? COLOUR_MAP[i] : error("Unknown colour id, expected one of $(keys(COLOUR_MAP)), got $(i)")
pmap(i) = (i in keys(COLOUR_MAP)) ? Symbol(COLOUR_MAP[i], :s) : error("Unknown colour id, expected one of $(keys(COLOUR_MAP)), got $(i)")

const STYLE_MAP = Dict(
    1 => :solid,
    2 => :dash,
    3 => :dot,
)
smap(i) = (i in keys(STYLE_MAP)) ? STYLE_MAP[i] : error("Unknown style id, expected one of $(keys(STYLE_MAP)), got $(i)")
