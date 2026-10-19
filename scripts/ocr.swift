import Foundation
import Vision

guard CommandLine.arguments.count >= 2 else {
    fputs("Usage: ocr <image>\n", stderr)
    exit(1)
}

let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])


// ---------------------------------------------------------
// Vision OCR
// ---------------------------------------------------------

let request = VNRecognizeTextRequest()

request.recognitionLevel = .accurate
request.usesLanguageCorrection = false

let supported =
    try VNRecognizeTextRequest().supportedRecognitionLanguages()

var languages: [String] = []

if supported.contains("en-US") {
    languages.append("en-US")
}

if supported.contains("ru-RU") {
    languages.append("ru-RU")
}

if !languages.isEmpty {
    request.recognitionLanguages = languages
}


let handler = VNImageRequestHandler(
    url: imageURL,
    options: [:]
)

do {
    try handler.perform([request])
} catch {
    fputs("Vision error: \(error)\n", stderr)
    exit(2)
}


guard let observations = request.results,
      !observations.isEmpty else {
    exit(0)
}


// ---------------------------------------------------------
// OCR segment
// ---------------------------------------------------------

struct Segment {
    var text: String
    var x: CGFloat
    var y: CGFloat
    var width: CGFloat
    var height: CGFloat
}


// ---------------------------------------------------------
// Convert Vision observations
// ---------------------------------------------------------

var segments: [Segment] = []

for observation in observations {

    guard let candidate =
        observation.topCandidates(1).first else {
        continue
    }

    let box = observation.boundingBox

    segments.append(
        Segment(
            text: candidate.string,
            x: box.minX,
            y: box.midY,
            width: box.width,
            height: box.height
        )
    )
}


// ---------------------------------------------------------
// Median helper
// ---------------------------------------------------------

func median(_ values: [CGFloat]) -> CGFloat {

    guard !values.isEmpty else {
        return 0
    }

    let sorted = values.sorted()

    return sorted[sorted.count / 2]
}


// ---------------------------------------------------------
// Estimate text line height
// ---------------------------------------------------------

let medianHeight =
    median(
        segments.map { $0.height }
    )


// ---------------------------------------------------------
// Estimate monospace character width
// ---------------------------------------------------------

var widthSamples: [CGFloat] = []

for segment in segments {

    let count = segment.text.count

    if count > 0 {

        let value =
            segment.width / CGFloat(count)

        if value > 0 {
            widthSamples.append(value)
        }
    }
}

var charWidth = median(widthSamples)

if charWidth <= 0 {
    charWidth = 0.01
}


// ---------------------------------------------------------
// Sort segments from top to bottom
// ---------------------------------------------------------

segments.sort { $0.y > $1.y }


// ---------------------------------------------------------
// GROUP SEGMENTS INTO VISUAL ROWS
//
// This is the important change.
//
// Vision may return:
//
//   [ n_workers=2, ]        [ # 4 GPU ]
//
// as two observations.
//
// We want them in ONE logical row.
// ---------------------------------------------------------

struct OCRRow {
    var segments: [Segment]
    var centerY: CGFloat
}


var rows: [OCRRow] = []

// Large enough for comment text whose bounding box
// is vertically shifted slightly.
let rowTolerance =
    max(
        medianHeight * 0.70,
        0.008
    )


for segment in segments {

    var added = false

    // Normally the matching row will be one of the last ones,
    // but searching all rows makes it more robust.
    for i in rows.indices {

        if abs(rows[i].centerY - segment.y) <= rowTolerance {

            rows[i].segments.append(segment)

            let count =
                CGFloat(rows[i].segments.count)

            rows[i].centerY =
                (
                    rows[i].centerY * (count - 1)
                    + segment.y
                ) / count

            added = true
            break
        }
    }


    if !added {

        rows.append(
            OCRRow(
                segments: [segment],
                centerY: segment.y
            )
        )
    }
}


// ---------------------------------------------------------
// Sort rows top -> bottom
// ---------------------------------------------------------

rows.sort {
    $0.centerY > $1.centerY
}


// ---------------------------------------------------------
// Sort each row left -> right
// ---------------------------------------------------------

for i in rows.indices {

    rows[i].segments.sort {
        $0.x < $1.x
    }
}


// ---------------------------------------------------------
// Find global code left margin
// ---------------------------------------------------------

let leftMargin =
    rows
        .flatMap { $0.segments }
        .map { $0.x }
        .min() ?? 0


// ---------------------------------------------------------
// Reconstruct rows
// ---------------------------------------------------------

var result: [String] = []


for row in rows {

    guard let first = row.segments.first else {
        continue
    }


    // -----------------------------------------------------
    // Leading indentation
    // -----------------------------------------------------

    let rawLeadingColumns =
        Int(
            round(
                (first.x - leftMargin)
                / charWidth
            )
        )


    let leadingColumns =
        max(0, rawLeadingColumns)


    var line =
        String(
            repeating: " ",
            count: leadingColumns
        )


    // -----------------------------------------------------
    // Reconstruct every OCR fragment on this visual row
    // -----------------------------------------------------

    for (index, segment) in row.segments.enumerated() {

        if index > 0 {

            let previous = row.segments[index - 1]
            let gap = segment.x - (previous.x + previous.width)
            let spaces = max(0, Int(round(gap / charWidth)))

            if spaces > 0 {
                line += String(repeating: " ", count: spaces)
            }
        }


        line += segment.text
    }


    result.append(
        line.trimmingCharacters(
            in: .whitespacesAndNewlines
        ).isEmpty
            ? ""
            : line
    )
}


// ---------------------------------------------------------
// Output
// ---------------------------------------------------------

print(
    result.joined(
        separator: "\n"
    )
)
