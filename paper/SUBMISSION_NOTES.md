# Eurographics 2027 -- submission notes

Working title: *Synthetic LiDAR-Camera Data for Human Pose and Shape Estimation*.
Facts below were collected on 2026-09-10 from the sources listed; re-check SRMv2 before submitting.

## Deadlines (EG 2027 Full Papers)

Source: <https://eg2027.isti.cnr.it/call-for-papers/>

| Milestone | Date |
|---|---|
| Abstract and submission form | **Fri 25 Sep 2026** |
| Full paper | **Thu 1 Oct 2026** |
| Reviews released to authors | Mon 23 Nov 2026 |
| Rebuttal due | Mon 30 Nov 2026 |
| Notification (conditional accept / reject) | Fri 18 Dec 2026 |
| Revised version due | Tue 26 Jan 2027 |
| Final notification | Tue 9 Feb 2027 |
| Camera-ready | Tue 23 Feb 2027 |
| Conference | 10-14 May 2027, Lucca, Italy (at least one author must register and present in person) |

Submission system: <https://srmv2.eg.org/COMFy/Conference/EG_2027>

## Page limit

- CfP page: "There is no maximum length imposed on papers. However, papers should only be as long as they need to be, but not longer. Reviewers might rank submissions perceived as being either unnecessarily long or too short lower."
- SRMv2 author instructions (<https://srmv2.eg.org/COMFy/Conference/EG_2027/Instruction>): "It is recommended that research papers be up to **10 pages** (in CGF latex style including all images but **excluding references**), and survey papers be up to 20 pages (excluding references)."
- Same recommendation on the EG publication guidelines page: <https://www.eg.org/wp/eurographics-publications/guidelines/>

Plan for 10 pages + references.

## Anonymity / double-blind

Source: CfP page and SRMv2 instructions.

- Two-step double-blind review. Remove all personal data (authors, affiliations, acknowledgements, project URLs that identify you).
- Use the SUBMISSION ID in place of author names (`\author[SUBMISSION ID]{SUBMISSION ID}` in the template).
- Cite your own prior work in the third person. Non-peer-reviewed prior work (arXiv, tech reports) that would identify you should *not* be cited; list it instead in the "Prepublications" field of the SRM form (visible to IPC only).
- Supplementary material must also be anonymised.
- SIGGRAPH-style anonymity policy: do not reveal identity in any submitted material, do not post/advertise the submission on social media during review; unlisted YouTube videos and anonymous code/data repositories are allowed.
- Accepted papers are published in a special issue of *Computer Graphics Forum* (Wiley) and the EG Digital Library.

## Format / LaTeX style

- Papers "must be formatted according to the Eurographics Computer Graphics Forum guidelines. The publication guidelines and LaTeX templates are available on SRMv2." (CfP)
- Class: `egpubl` ("EG publication style", `egpubl.cls`, currently v3.96 of 2024-09-24).
- Official downloads:
  - EG2027-specific package on SRMv2 (requires an SRMv2 login): <https://srmv2.eg.org/COMFy/Conference/EG_2027/GetConferenceFile?fileID=20175>.
    The instructions say to use `EGauthorGuidelines-conf-sub.tex` from that zip and, for a teaser, uncomment the `\teaser` block in `EGauthorGuidelines-body-sub.inc`.
    `EGauthorGuidelines-conf-fin_with_teaser.tex` is a Biber example and will not compile with BibTeX.
  - Public CGF package (same `egpubl.cls`) on eg.org: <https://www.eg.org/wp/wp-content/uploads/2024/12/egPublStyle-cgf.zip>
    (linked from <https://www.eg.org/wp/eurographics-publications/guidelines/>). This is what is vendored in `eg-style/`.
- Status: **downloaded**. `curl` against the SRMv2 link only returned the login page, so `eg-style/` contains the public eg.org
  `egPublStyle-cgf` package instead. `main.tex` follows its submission preamble (`EGauthorGuidelines-cgf-sub.tex`) with `\JournalSubmission`.
  TODO before submitting: log in to SRMv2, download the EG_2027 package, and diff `egpubl.cls` / the `conf-sub` preamble against ours
  (the conference-issue template may select a different mode switch such as `\SpecialIssueSubmission`).
- Bibliography: `eg-alpha-doi.bst` (BibTeX) as in the template; alternatively biblatex with `EG-sub.bbx` for submission (keeps full author lists for conflict checking).
- Fonts/packages required by the class: `dfadobe.sty` (shipped), `cite`, `url`, `hyperref`, `ifpdf`, `lastpage`.

## Supplementary material

Source: SRMv2 instructions.

- Allowed and encouraged: videos, high-resolution images, appendices with derivations, raw/processed data, source code. Refer to it from the paper; reviewers are encouraged to look at it.
- Main paper must be PDF. Total upload (paper + additional materials + representative image) <= **500 MB**.
- Optional cover letter (as "Additional Attachment") is "strongly encouraged" for resubmissions/prior reviews.
- Graphics Replicability Stamp available for open-source implementations (<http://www.replicabilitystamp.org/>).

## Other

- AI-generated content: CGF rules apply; any GenAI use in preparing the manuscript must be declared in the AI Use Declaration.
- Best paper: Guenter Enderle Award; Best-of-Eurographics session at SIGGRAPH 2027.
