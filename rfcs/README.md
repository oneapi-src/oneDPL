# oneDPL Design Documents/RFCs

The Request for Comments (RFC) process intends to:

- Propose and discuss ideas for the library evolution.
- Communicate anticipated library-wide changes.
- Collect feedback before implementation.
- Increase transparency in decision-making.
- Align different teams involved in the library development.

Most modifications or new features will naturally start as a part of a 
GitHub issue or discussion. Small changes do not require a formal RFC. 
However, if the issue or discussion results in an idea for a significant 
change or new feature that affects the library's public API or architecture, 
we recommend creating a formal document that provides
a detailed description and design of the proposed feature.

This directory contains design documents (RFCs) approved, 
implemented, or rejected for implementation.

## RFC Directory Structure

The design documents are stored in the `rfcs` directory, each placed 
in a subdirectory under an `rfcs/<state>` directory. The possible states are:

1. Proposed
2. Experimental
3. Supported
4. Archived

The `rfcs/proposed` directory contains RFCs for approved proposals
that need to be implemented. These documents describe the overall design
and API for the proposed functionality.

The `rfcs/experimental` directory contains RFCs for experimental library features.
In addition to the design, these documents describe the criteria for the described
functionality to exit the experimental status.

The `rfcs/supported` directory contains documents for the fully supported features,
both implemented according to the library specification and provided as extensions.

The `rfcs/archived` directory contains rejected proposals and documents for
the former functionality that has been removed.

A subdirectory for an RFC should have a name of the form `<library_feature>_<extension_description>`
and should contain a `README.md` file that either is the RFC document
or links to other files and Web resources that describe the functionality.
The directory can contain other supporting files such as images or formulas,
as well as sub-proposals / sub-RFCs.

## General Process

You can collect initial feedback on an idea and input for a formal RFC proposal
using a GitHub discussion. Add the "RFC" label to the discussion to indicate
the intent. The discussion can also be used to reference relevant information
and keep track of the progress.

To create a new RFC document, open a pull request (PR) to add it to the `rfcs/proposed` directory.
A template for new RFCs is available as [template.md](template.md).
Use it to create the `README.md` file in a subdirectory of `rfcs/proposed` named
`<library_feature>_<extension_description>`. For example,
a proposal for adding a bitonic sorting algorithm working with C++ ranges would be put
into the `rfcs/proposed/range_algorithms_bitonic_sort` directory.
Put other files referenced by the `README.md` file, such as figures, into the same directory.
The "RFC" label can be used to mark PRs containing RFC/design proposals.

The RFC approval process generally follows the guidelines in the [UXL Foundation Operational Procedures](
https://github.com/uxlfoundation/uxl_operational_procedures/blob/release/Process_Documents/Organization_Operational_Process.md#review--approval-process).
Once two or more maintainers approve the PR, it is merged into the main branch
as an RFC proposed for implementation.

As the RFC moves to different states, use new PRs to update the RFC document
with additional information.

A proposal that is subsequently implemented and released as an experimental feature
is moved into the `rfcs/experimental` folder.
The RFC for such a feature should include a description
of what is required to move it from experimental to fully supported -- for 
example, feedback from users, demonstrated performance improvements, etc.

A proposal that is implemented as a fully supported feature appears
in the `rfcs/supported` directory. It typically involves the oneDPL specification
changes and should therefore have a link to the section in the specification
with its formal wording.

A feature that is removed or a proposal that is abandoned or rejected will 
be moved to the `rfcs/archived` folder. It should state the reasons for
rejection or removal.

## Document Style Recommendations

- Follow the document structure described in [template.md](template.md).
- We highly recommend using a text-based file format like markdown for easy 
collaboration on GitHub, but other formats like PDFs may also be acceptable.
- For the markdown-written RFC, keep the text width within 100 characters,
unless there is a reason to violate this rule, e.g., long links or wide tables.
- It is also recommended to read through existing RFCs to better understand the 
general writing style and required elements.
