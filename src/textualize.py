def textualize_prevalence(prevalence: dict):
    prevalence_description = []
    if prevalence["PrevalenceClass"]:
        prevalence_description.append(f"PrevalenceClass: {prevalence['PrevalenceClass']}")
    if prevalence["PrevalenceGeographic"]:
        prevalence_description.append(f"PrevalenceGeographic: {prevalence['PrevalenceGeographic']}")
    if prevalence["PrevalenceQualification"]:
        prevalence_description.append(f"PrevalenceQualification: {prevalence['PrevalenceQualification']}")
    if prevalence["PrevalenceValidationStatus"]:
        prevalence_description.append(f"PrevalenceValidationStatus: {prevalence['PrevalenceValidationStatus']}")
    if prevalence["Source"]:
        prevalence_description.append(f"Source: {prevalence['Source']}")
    if prevalence["ValMoy"]:
        prevalence_description.append(f"ValMoy: {prevalence['ValMoy']}")
    return prevalence_description


def textualize_organization(organization: dict):
    organization_description = [organization["Name"]]
    if organization["Address1"] or organization["Address2"]:
        organization_description.append(f"Address: ")
        organization_description.append(f"{organization['Address1']}" if organization["Address1"] else "")
        organization_description.append(f"{organization['Address2']}" if organization["Address2"] else "")
    if organization["City"]:
        organization_description.append(f"City: {organization['City']}")
    if organization["Country"]:
        organization_description.append(f"Country: {organization['Country']}")
    if organization["Email"]:
        organization_description.append(f"Email: {organization['Email']}")
    if organization["Fax"]:
        organization_description.append(f"Fax: {organization['Fax']}")
    if organization["Phone"]:
        organization_description.append(f"Phone: {organization['Phone']}")
    if organization["State"]:
        organization_description.append(f"State: {organization['State']}")
    if organization["TollFree"]:
        organization_description.append(f"TollFree: {organization['TollFree']}")
    if organization["URL"]:
        organization_description.append(f"URL: {organization['URL']}")
    if organization["ZipCode"]:
        organization_description.append(f"ZipCode: {organization['ZipCode']}")
    return organization_description
